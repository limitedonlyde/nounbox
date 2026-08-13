# Хостинг всей платформы на Modal — проектный документ

> Режим «нет своего сервера — одна команда, и всё крутится на Modal».
> **Дополнение** к docker-compose (основной self-hosted путь), не замена.
> Видение: ядро лёгкое и запускается где угодно; тяжёлый GPU-инференс уже вынесен
> на Modal (`deploy/modal/vlm.py`, `deploy/modal/paddleocr_modal.py`) — здесь
> проектируем вынос самого ядра (API + worker + БД + файлы + UI).

Статус: дизайн (код не менялся). Дата исследования: 2026-08-11.

---

## 0. Врезка: GPU-рецепт в автодеплое из UI

Отдельная от этого документа, уже принятая владельцем схема: разметка живёт в двух
режимах — **ПРОСТО** (`rapidocr`, CPU, работает сразу после `docker compose up`) и
**СЛОЖНО** (`modal_gpu`, по желанию). Второй режим платформа поднимает сама, а
`paddleocr_modal.py` — тот самый рецепт, который она деплоит.

Поток: пользователь вписывает на странице настроек API-токен Modal
(`ak-…` / `as-…`, то что даёт `modal token new`; `PUT /api/v1/settings`) и жмёт
«Подключить GPU» (`POST /api/v1/settings/gpu/deploy` → Job типа `DEPLOY_GPU`).
Платформа импортирует `deploy/modal/paddleocr_modal.py`, берёт объект `app` с уровня
модуля и вызывает `modal.runner.deploy_app(app, name="autolabelui-paddleocr",
client=modal.Client.from_credentials(...))`; URL достаёт отдельным
`modal.Function.from_name(app, "fastapi_app", client=…).hydrate(client=…).get_web_url()`
(в `DeployResult` его нет) и кладёт в `settings.gpu_endpoint_url`. Дальше worker в
`run_autolabel` сам подставляет endpoint в конфиг движка `modal_gpu` — руками JSON
никто не пишет.

Что из этого следует для рецепта (и уже сделано):
- `app` на уровне модуля, импорт без побочных эффектов и без чтения локальных файлов —
  деплой идёт из FastAPI-процесса, а не из CLI;
- файл самодостаточен: `include_source=True` монтирует в контейнер Modal РОВНО этот
  один `.py`, любой `from ...` из монорепо там не разрешится;
- токен доступа: если у процесса-деплойщика задан `AUTOLABELUI_GPU_TOKEN`, он
  запекается в Secret приложения (`Secret.from_dict`, читается при импорте — значит
  выставлять переменную нужно ДО импорта модуля) и `/predict` начинает требовать
  `Authorization: Bearer <token>`; переменной нет — эндпоинт открыт по неугадываемому
  URL, как было;
- кнопка идемпотентна бесплатно: повторный деплой — новая версия (rolling), при
  отсутствии изменений Modal отвечает «Deployment skipped». `strategy="recreate"`
  не использовать — убьёт живые контейнеры.

Проверка после деплоя: `GET /health` отдаёт `{"auth": "bearer"|"open",
"engines_loaded": N}` — этим платформа убеждается, что контейнер поднялся на GPU и
что токен доехал. Первый `POST /predict` дополнительно платит за скачивание весов
PP-OCRv5 (кешируются в Volume `autolabelui-paddlex-cache`), поэтому «прогревать»
эндпоинт лучше крошечным PNG, а не первой реальной страницей пользователя.

---

## 1. Что даёт Modal (сводка исследования)

Факты из актуальной документации, на которые опирается дизайн:

**Web endpoints / ASGI.** `@modal.asgi_app()` отдаёт целое FastAPI-приложение как
web endpoint (`https://<workspace>--<app>-<fn>.modal.run`); `@modal.concurrent(max_inputs=N)`
позволяет одному контейнеру обслуживать N запросов параллельно (event loop ASGI).
Тело запроса до 4 GiB. Кастомные домены — только Team/Enterprise план; на Starter
остаёмся на `*.modal.run`. Защита endpoint'а — Proxy Auth Tokens
(`requires_proxy_auth=True`, заголовки `Modal-Key`/`Modal-Secret` или
`Authorization: Bearer <id>.<secret>`), но браузерному UI такие заголовки не
подставить — для UI нужен собственный токен-механизм (см. §5).

**Cold start.** Boot контейнера ~1 с; реальную задержку даёт инициализация
приложения (импорты, модели). Инструменты: `scaledown_window` (2 с … 20 мин,
дефолт 60 с), `min_containers` (не опускаться до нуля — но платим за idle),
`buffer_containers`, memory snapshots (снимок прогретой памяти — «substantially
reduce cold start latency»).

**Volumes.** Распределённая ФС с коммитами: ручной `.commit()` + фоновые
автокоммиты «каждые несколько секунд» и финальный снапшот при остановке.
Семантика — **last write wins по файлам**; «concurrent modifications of the same
files should be avoided»; **file locking не поддерживается** («nor is distributed
file locking supported»). Пропускная способность до 2,5 GB/s, деградация при
>50 000 файлов (v1), Volumes v2 масштабируются лучше, но правило «один писатель
на файл» остаётся. **Вывод для СУБД:** любая база на Volume допустима только при
строго одном контейнере-писателе; атомарность коммита нескольких файлов
(например, `*.db` + `*-wal`) не гарантируется — нужны свои чекпоинты/бэкапы.
Это ключевое ограничение всего дизайна.

**Sandboxes.** Контейнеры, создаваемые в рантайме, для недоверенного кода: max
lifetime 24 ч, эфемерные, есть tunnels и volume-mount. Как «дом» для платформы
не годятся (ограничение времени жизни, ручной lifecycle), но это готовый
механизм для будущего запуска **сторонних labeler-плагинов в изоляции** (v0.3+).

**Dict / Queue.** Queue: элемент ≤ 1 MiB, ≤ 5 000 элементов на партицию,
партиция очищается через 24 ч после последнего put. Годится как транспорт
очереди, но проще и надёжнее `Function.spawn()`: возвращает `FunctionCall`,
результат опрашивается по id до 7 дней, есть `retries`/`timeout` на функции.
Dict пригодится для progress-репортинга длинных джобов (не критично для v1).

**Auto Endpoints (managed).** `modal endpoint create --model <hf-id>` — управляемый
OpenAI-совместимый endpoint (`/v1`), scale-to-zero, оплата только за компьют,
код рецепта открыт. Каталог: семейства Qwen, Kimi, Gemma, DeepSeek, Nemotron,
GPT-OSS, GLM. **Vision-модели (Qwen2.5-VL / Qwen3-VL) в каталоге документацией
не подтверждены** (в блоге упоминается «vision-language model Endpoint», но
конкретных VL-моделей в списке нет). Значит для VLM-labeler'а остаётся наш
`deploy/modal/vlm.py` (vLLM + Qwen2.5-VL); стоит периодически проверять каталог —
как только VL-модели появятся, рецепт можно заменить одной командой. Официальный
пример Modal с SGLang + Qwen-VL — запасной вариант рецепта.

**Цены (актуальные).** CPU $0.0000131/физ.ядро/с (минимум 0.125 ядра),
RAM $0.00000222/GiB/с, GPU: T4 $0.000164/с (≈$0.59/ч), L4 $0.000222/с (≈$0.80/ч),
A100-80GB $0.000694/с, H100 $0.001097/с. Volume $0.09/GiB/мес (с включённой
квотой). Starter-план: **$30/мес бесплатных кредитов**, оплата только за
фактическое время работы контейнеров (idle при scale-to-zero не тарифицируется;
при `min_containers=1` — тарифицируется).

---

## 2. Текущая привязка кода к инфраструктуре

Что именно держит ядро на docker-compose (по файлам):

| Зависимость | Где в коде | Что мешает Modal-режиму |
|---|---|---|
| Postgres (asyncpg) | `server/app/db.py`, `config.py: database_url` | нужен работающий Postgres |
| Диалект-специфика PG | `server/app/models.py` (`JSONB`, `UUID` из `sqlalchemy.dialects.postgresql`), `workers/tasks.py:70` (`.astext`) | не заведётся на SQLite |
| MinIO/S3 | `server/app/storage.py` (boto3, presigned URL) | нужен S3-сервис |
| Redis + arq | `main.py` (`create_pool`), `api/documents.py:50`, `api/jobs.py:32` (`enqueue_job`), `workers/tasks.py` (`WorkerSettings`) | нужен Redis и отдельный процесс-воркер |
| UI как отдельный dev-сервер | `web/` (Vite, прокси на api) | нужен второй HTTP-сервис |

Сами задачи (`run_ingest`, `run_autolabel`) — обычные async-функции без
привязки к arq, кроме сигнатуры `(ctx, job_id)`: переносимы как есть.

---

## 3. Вариант A — «Modal-native lite»: ASGI + SQLite + Volume + spawn

### Архитектура

Один Modal App `autolabelui-platform`, один класс-сервис, **один контейнер**:

```python
# deploy/modal/platform.py (эскиз)
data_vol = modal.Volume.from_name("autolabelui-data", create_if_missing=True)

@app.cls(
    image=image,                      # server + sdk + labelers/http,vlm + web/dist
    volumes={"/data": data_vol},
    max_containers=1,                 # ЕДИНСТВЕННЫЙ писатель SQLite и файлов
    scaledown_window=1200,            # 20 мин — максимум
    timeout=3600,
)
@modal.concurrent(max_inputs=32)
class Platform:
    @modal.asgi_app()
    def web(self):
        from app.main import app as fastapi_app   # env: MODE=modal
        return fastapi_app

    @modal.method()
    async def run_job(self, task: str, job_id: str):
        from app.workers import tasks
        await getattr(tasks, task)({}, job_id)
```

- **БД**: SQLite на Volume (`sqlite+aiosqlite:////data/autolabel.db`).
- **Файлы**: тот же Volume (`/data/objects/...`) вместо MinIO.
- **Очередь**: `Platform().run_job.spawn("run_ingest", job_id)` вместо
  `arq.enqueue_job`. При `max_containers=1` spawn попадает в **тот же контейнер
  и процесс** → тот же SQLite-коннект, и — важно — пока джоб выполняется, он
  считается активным input'ом: контейнер не скейлится в ноль посреди ingest'а.
- **UI**: `web/dist` (vite build) раздаётся самим FastAPI через `StaticFiles` —
  один endpoint на всё.
- **GPU-инференс**: без изменений — HTTP/VLM-labeler'ы ходят в отдельные
  Modal-приложения (`vlm.py`, `paddleocr_modal.py`) или в Auto Endpoint.

### Необходимые изменения кода (все полезны и для compose)

1. `server/app/models.py` — портируемые типы:
   `JSONB` → `JSON().with_variant(JSONB, "postgresql")`;
   `UUID(as_uuid=True)` (диалект PG) → `sqlalchemy.Uuid` (SA 2.0, на SQLite —
   CHAR(32), на PG — родной uuid). Поведение на Postgres не меняется.
2. `server/app/workers/tasks.py:70` — `Annotation.source["name"].astext` →
   `.as_string()` (портируемый аксессор; на PG компилируется в `->>`).
3. `server/app/config.py` — `storage_backend: s3|local`, `local_storage_dir`,
   `queue_backend: arq|modal`, `serve_static: bool`.
4. `server/app/storage.py` — интерфейс с двумя реализациями: текущая S3 и
   `LocalStorage` (файлы в каталоге на Volume; `presigned_url` → подписанный
   HMAC-токеном роут `GET /api/v1/files/{key}?exp=&sig=`). Один новый роут.
5. Новый `server/app/queue.py` — `async def enqueue(request, task, job_id)`:
   arq-реализация (текущее поведение) и modal-реализация
   (`modal.Cls.from_name("autolabelui-platform", "Platform")().run_job.spawn(...)`).
   `main.py` создаёт arq-pool только при `queue_backend=arq`.
6. `server/app/db.py` — для SQLite: `PRAGMA journal_mode=WAL`,
   `busy_timeout`, NullPool; `pyproject.toml` + `aiosqlite`.
7. Новый `deploy/modal/platform.py` (~100 строк) — эскиз выше.

### Свойства

- **Cold start**: ~1 с контейнер + 2–4 с импорты FastAPI/SQLAlchemy; с memory
  snapshot — заметно меньше. При желании `min_containers=1` убирает cold start
  совсем (см. цену). После 20 мин тишины первый запрос ждёт несколько секунд —
  для инструмента разметки приемлемо.
- **Персистентность**: SQLite и файлы переживают рестарты через коммиты Volume.
  Риск: фоновый коммит не атомарен между `*.db` и `*-wal` — митигируем
  (а) `PRAGMA wal_checkpoint(TRUNCATE)` после каждого джоба и по таймеру,
  (б) периодический `VACUUM INTO /data/backup/autolabel-<ts>.db` (целостная
  копия одним файлом), (в) экспортные ZIP — и есть офсайт-бэкап датасета.
  При крахе между коммитами теряются последние секунды записи — для малой
  нагрузки терпимо, для команды — уже нет (тогда вариант C).
- **Конкурентность**: `@modal.concurrent` даёт параллельные запросы в одном
  процессе; SQLite WAL спокойно держит 1–5 пользователей ревью. Горизонтального
  масштабирования нет by design (`max_containers=1`) — это осознанный потолок.
- **Цена/мес при малой нагрузке** (scale-to-zero, ~60 ч активности/мес,
  0.5 ядра, 1 GiB): CPU 216 000 с × 0.5 × $0.0000131 ≈ $1.4 + RAM ≈ $0.5 ≈
  **$2/мес**; с `min_containers=1` (0.125 ядра, ~0.75 GiB круглосуточно) ≈
  $4.2 + $4.3 ≈ **$8.6/мес**. Volume — в пределах включённой квоты. Всё
  покрывается $30 бесплатных кредитов → **фактически $0**. GPU — отдельно,
  по факту разметки ($0.59–0.80/ч T4/L4).
- **Сложность**: средняя — 6 правок + 1 новый файл, ~2–4 дня с e2e-проверкой.

---

## 4. Вариант B — «всё в одном контейнере»: supervisor(postgres+minio+redis+api+worker) + Volume

### Архитектура

Один образ с supervisord/s6: postgres, redis, minio, uvicorn, arq-worker;
`PGDATA`, данные MinIO и дампы — на Volume; наружу — `@modal.web_server` на порт
API. Обязательно `max_containers=1` и фактически `min_containers=1`.

### Изменения кода

Почти нулевые: те же env-переменные, что в compose (`database_url` на
localhost и т.д.). Работа уходит в образ: установка postgres/minio/redis,
supervisor-конфиг, скрипт первичной инициализации, healthcheck-и, порядок
старта. Плюс раздача `web/dist` (как в A, п. StaticFiles).

### Честно про Postgres на Modal Volumes

Это самое слабое место варианта:

- Postgres постоянно перезаписывает множество файлов (heap, WAL, pg_control).
  Фоновый коммит Volume снимает состояние «каждые несколько секунд» **без
  атомарности между файлами** — восстановленный после краха `PGDATA` может быть
  рассинхронизован (WAL от одного момента, heap от другого). Это классический
  сценарий «looks fine until it doesn't»: Postgres может подняться и молча
  потерять/побить данные, а может не подняться вовсе.
- Документация Modal прямо не запрещает СУБД, но её модель («avoid concurrent
  modifications of the same files», нет file locking, last-write-wins)
  ориентирована на write-once артефакты (веса моделей, датасеты), не на
  data-каталог СУБД.
- Scale-to-zero недопустим (остановка = финальный снапшот горячего PGDATA),
  значит контейнер 24/7 и вся экономика serverless теряется.
- Митигация только одна — относиться к PGDATA как к кешу: `pg_dump` каждые
  N минут на Volume/в S3 и готовность восстанавливаться из дампа. Тогда зачем
  вообще живой Postgres?

### Свойства

- **Cold start**: не применим (24/7); рестарт — 10–30 с (recovery Postgres).
- **Персистентность**: формально есть, фактически — риск коррапта PGDATA при
  любом недобровольном рестарте; надёжна только связка «PGDATA-как-кеш + частый
  pg_dump».
- **Конкурентность**: внутри контейнера — полноценный Postgres, но предел один
  контейнер.
- **Цена/мес**: 24/7, 1 ядро + 2 GiB: $34 + $11.5 ≈ **$45/мес** (0.5 ядра +
  1.5 GiB ≈ $26/мес) — дороже VPS с теми же гарантиями.
- **Сложность**: по коду низкая, по образу/эксплуатации — высокая; постоянный
  операционный риск.

**Вердикт**: анти-паттерн для Modal. Полезен разве что как throwaway-демо.

---

## 5. Вариант C — гибрид: ядро на Modal, состояние во внешних managed-сервисах

### Архитектура

- API+worker — как в A (тот же `platform.py`, но `max_containers` можно > 1).
- **Postgres — Neon** (serverless, free tier: 0.5 GB, 100 CU-часов/мес,
  autosuspend через 5 мин, авто-resume ~сотни мс). `postgresql+asyncpg://…` —
  **текущий код работает без изменений**, JSONB/UUID остаются родными.
- **Файлы — Cloudflare R2** (S3-совместимый, free tier 10 GB, **нулевой
  egress**) или любой S3. `storage.py` работает как есть: меняются только
  `s3_endpoint_url`/ключи; presigned URL на R2 поддерживаются.
- **Очередь — Modal spawn** (как в A; Redis не нужен вовсе) — либо Upstash
  Redis + текущий arq, но это лишний сервис и постоянный поллинг воркера.

### Изменения кода

Если сделан вариант A — **ноль**: C это конфигурация A
(`database_url=neon, storage_backend=s3(r2), queue_backend=modal`).
Без A: только queue-абстракция (п. 5 из §3) + раздача статики.

### Свойства

- **Cold start**: как A (+~0.5–1 с на resume Neon после простоя).
- **Персистентность**: лучшая из трёх — настоящий Postgres с бэкапами Neon,
  объекты в R2 с их durability. Никаких компромиссов с Volume-семантикой.
- **Конкурентность**: лучшая — API стателесс, можно `max_containers>1`,
  джобы — параллельные spawn'ы, Postgres честно держит конкурентные записи.
  Это же путь к multi-user (v0.4).
- **Цена/мес при малой нагрузке**: compute как в A ($0–9, покрыто кредитами) +
  Neon Free $0 + R2 Free $0 ≈ **$0/мес**; рост — плавный и предсказуемый.
- **Минусы**: **три аккаунта вместо одного** (Modal + Neon + Cloudflare) —
  ломает «одна команда и всё крутится»; данные разъезжаются по трём
  провайдерам (для приватных датасетов кому-то это стоп-фактор); free tier'ы
  меняются, появляется зависимость от чужих лимитов.
- **Сложность**: поверх A — тривиальная (документация + пример .env).

---

## 6. Сравнение

| Критерий | A: SQLite+Volume | B: supervisor+PG | C: гибрид Neon+R2 |
|---|---|---|---|
| «Одна команда» без прочих аккаунтов | ✅ `modal deploy` | ✅ но долго | ❌ 3 аккаунта |
| Изменения кода | средние (портируемость) | ~нет (всё в образ) | 0 поверх A |
| Cold start | 3–5 с (0 при min_containers) | нет (24/7) | как A + Neon resume |
| Целостность данных | ок с чекпоинтами; риск на крахе | ⚠️ реальный риск PGDATA | ✅ лучшая |
| Конкурентность | 1 контейнер, WAL | 1 контейнер, PG | ✅ горизонтальная |
| Цена/мес (малая нагрузка) | ~$2–9 → $0 с кредитами | ~$26–45 | ~$0 |
| Путь к multi-user (v0.4) | упирается в 1 контейнер | упирается | ✅ готов |
| Сложность | средняя | образ+эксплуатация | +docs к A |

---

## 7. Рекомендация

**Делать вариант A как режим по умолчанию, спроектированный так, что C — это
его конфигурация.** Вариант B отклонить.

Обоснование:

1. Только A даёт обещанное владельцем «нет сервера — одна команда»: весь стейт
   в Modal, никаких сторонних регистраций, $30 кредитов покрывают всё.
2. 90% работ по A (портируемые типы, storage/queue-абстракции, статика) — это
   не «код под Modal», а снятие жёсткой привязки ядра к compose-стеку. Это
   прямо усиливает видение «ядро запускается на любом ПК»: side-effect'ом
   получаем режим `docker run` одним контейнером с SQLite без compose вообще.
3. C автоматически становится документированным upgrade-путём, когда пользователю
   станет тесно (команда, большие датасеты, multi-user в v0.4): поменять три
   env-переменных, перелить данные — код тот же. Не нужно выбирать между A и C
   сейчас — A их обоих содержит.
4. B противоречит семантике Modal Volumes (нет атомарных мульти-файловых
   коммитов, нет локов) и экономике serverless (24/7 контейнер дороже VPS).

Ограничение A принимаем осознанно: один контейнер, малые команды. Порог, после
которого пора на C, фиксируем в доке (примерно: >3–5 активных ревьюеров или
БД > 1–2 GB).

---

## 8. Поэтапный план внедрения (вариант A → C)

**Этап 0. Портируемость ядра (не меняет поведения compose).**
- `models.py`: `JSON().with_variant(JSONB, "postgresql")`, `sqlalchemy.Uuid`;
  `tasks.py`: `.as_string()`.
- `storage.py` → backend-интерфейс (S3 + Local с подписанным file-роутом).
- Новый `queue.py` (arq | modal), `main.py` — условное создание arq-pool.
- `config.py`: `storage_backend`, `queue_backend`, `local_storage_dir`,
  `serve_static`; `aiosqlite` в зависимости.
- Проверка: e2e на compose зелёный (ingest → autolabel → review → export);
  smoke на `sqlite+aiosqlite` локально.

**Этап 1. `deploy/modal/platform.py`.**
- Образ: server + sdk + `labelers/http` + `labelers/vlm` (без paddle-плагина —
  тяжёлый CPU-инференс в лёгком ядре не нужен, есть GPU-рецепт) + `web/dist`.
- Класс `Platform` (эскиз в §3): `max_containers=1`, `@modal.concurrent`,
  volume `/data`, `web()` + `run_job()`.
- SQLite: WAL, `busy_timeout`, `wal_checkpoint(TRUNCATE)` в конце джоба.
- Проверка: `modal deploy deploy/modal/platform.py` → полный e2e через
  `*.modal.run`, включая autolabel через `paddleocr_modal` endpoint.

**Этап 2. Прочность и UX.**
- Доступ: `AUTH_TOKEN` в Modal Secret → Bearer-проверка middleware'ом + поле
  токена в UI (одно на инсталляцию; browser-совместимая замена proxy-auth).
- Бэкапы: `VACUUM INTO /data/backup/` по завершении джобов (ротация N копий);
  README-раздел про `modal volume get` для эвакуации данных.
- Cold start: замерить; включить memory snapshot; задокументировать
  `min_containers=1` как платную опцию «без задержек».
- README: раздел «Хостинг на Modal» — буквально
  `pip install modal && modal setup && modal deploy deploy/modal/platform.py`.

**Этап 3. Upgrade-путь C (только конфиг + доки).**
- Пример `.env.modal-hybrid` (Neon `database_url`, R2 endpoint/ключи,
  `storage_backend=s3`), инструкция миграции SQLite → Postgres (alembic + скрипт
  переноса; синергия с уже запланированным переходом на Alembic-миграции).
- e2e на Neon free tier + R2.

**Этап 4. Наблюдение и развитие.**
- Периодически проверять каталог Auto Endpoints: появление Qwen2.5-VL/Qwen3-VL
  → заменить `vlm.py` на managed endpoint (меньше нашего кода).
- Прогресс джобов через `modal.Dict` (опционально).
- Sandboxes — кандидат для изоляции сторонних labeler-плагинов (v0.3+).

Ориентировочно: этап 0 — 1–2 дня, этап 1 — 1–2 дня, этап 2 — 1 день,
этап 3 — 0.5–1 день.

---

## 9. Источники

- Volumes (коммиты, конкурентность, отсутствие локов): <https://modal.com/docs/guide/volumes>
- Цены CPU/RAM/GPU/Volume, $30 кредитов Starter: <https://modal.com/pricing>
- Web endpoints, `@modal.asgi_app`, `@modal.concurrent`: <https://modal.com/docs/guide/webhooks>
- URL'ы endpoint'ов, кастомные домены (Team/Enterprise): <https://modal.com/docs/guide/webhook-urls>
- Proxy Auth Tokens: <https://modal.com/docs/guide/webhook-proxy-auth>
- Cold start: `min_containers`, `buffer_containers`, `scaledown_window`, memory snapshots: <https://modal.com/docs/guide/cold-start>
- Очередь джобов через `spawn`/`FunctionCall` (результаты 7 дней): <https://modal.com/docs/guide/job-queue>
- Dict и Queue (паттерны): <https://modal.com/docs/guide/dicts-and-queues>; лимиты Queue: <https://modal.com/docs/reference/modal.Queue>
- Sandboxes (lifecycle, 24 ч, tunnels): <https://modal.com/docs/guide/sandboxes>
- Auto Endpoints (каталог, `modal endpoint create`, OpenAI-совместимость): <https://modal.com/docs/guide/endpoints>; анонс: <https://modal.com/blog/introducing-auto-endpoints>
- Пример VLM (SGLang + Qwen-VL) на Modal: <https://modal.com/docs/examples/sglang_vlm>
- Neon free tier (0.5 GB, 100 CU-ч, autosuspend 5 мин): <https://neon.com/pricing>
- Cloudflare R2 (free tier 10 GB, нулевой egress): <https://developers.cloudflare.com/r2/pricing/>
