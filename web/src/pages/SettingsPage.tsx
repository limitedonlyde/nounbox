import { useCallback, useEffect, useRef, useState } from "react";
import { api, Settings } from "../api";

const STATUS_TEXT: Record<Settings["gpu_status"], string> = {
  not_configured: "GPU-движок не подключён",
  deploying: "Разворачивается в вашем аккаунте Modal...",
  ready: "GPU-движок готов",
  failed: "Деплой не удался",
};

function GpuStatusBlock({ settings }: { settings: Settings }) {
  const { gpu_status: status } = settings;
  return (
    <div className={`gpu-status gpu-status-${status}`}>
      <div className="gpu-status-title">
        {status === "deploying" && <span className="spinner" />}
        {STATUS_TEXT[status]}
      </div>
      {status === "deploying" && (
        <p className="muted">
          Первый деплой собирает образ с моделями — это может занять несколько
          минут. Страницу можно закрыть, процесс продолжится на сервере.
        </p>
      )}
      {status === "ready" && settings.gpu_endpoint_url && (
        <p className="muted">
          Endpoint:{" "}
          <a href={settings.gpu_endpoint_url} target="_blank" rel="noreferrer">
            {settings.gpu_endpoint_url}
          </a>
          <br />
          Движок <code>modal_gpu</code> доступен в селекторе на странице проекта.
        </p>
      )}
      {status === "failed" && settings.gpu_error && (
        <p className="error">{settings.gpu_error}</p>
      )}
    </div>
  );
}

function SettingsPage() {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [tokenId, setTokenId] = useState("");
  const [tokenSecret, setTokenSecret] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const pollRef = useRef<number | null>(null);
  const jobRef = useRef<string | null>(null);

  const stopPoll = useCallback(() => {
    if (pollRef.current !== null) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    jobRef.current = null;
  }, []);

  useEffect(() => stopPoll, [stopPoll]);

  const fail = useCallback((err: unknown) => {
    setError(err instanceof Error ? err.message : String(err));
  }, []);

  const startPoll = useCallback(() => {
    if (pollRef.current !== null) return;
    // деплой длится минуты: одиночный сетевой сбой не должен убивать опрос
    let misses = 0;
    pollRef.current = window.setInterval(async () => {
      try {
        if (jobRef.current !== null) {
          const job = await api.getJob(jobRef.current);
          if (job.status !== "done" && job.status !== "failed") return;
          jobRef.current = null;
        }
        const s = await api.getSettings();
        misses = 0;
        setSettings(s);
        setError(null);
        if (s.gpu_status !== "deploying") {
          stopPoll();
          setMessage(
            s.gpu_status === "ready"
              ? "GPU-движок развёрнут и готов к работе."
              : null
          );
        }
      } catch (err) {
        misses += 1;
        if (misses < 3) return;
        stopPoll();
        fail(err);
      }
    }, 3000);
  }, [fail, stopPoll]);

  const load = useCallback(async () => {
    try {
      const s = await api.getSettings();
      setSettings(s);
      setError(null);
      if (s.gpu_status === "deploying") {
        startPoll();
      }
    } catch (err) {
      fail(err);
    }
  }, [fail, startPoll]);

  useEffect(() => {
    void load();
  }, [load]);

  const saveToken = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!tokenId.trim() || !tokenSecret.trim()) return;
    setBusy(true);
    setMessage(null);
    setError(null);
    try {
      setSettings(await api.saveModalToken(tokenId.trim(), tokenSecret.trim()));
      setTokenId("");
      setTokenSecret("");
      setMessage("Токен сохранён.");
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  const disconnect = async () => {
    setBusy(true);
    setMessage(null);
    setError(null);
    stopPoll();
    try {
      setSettings(await api.deleteModalToken());
      setMessage("Токен удалён, GPU-режим отключён.");
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  const deploy = async () => {
    setBusy(true);
    setMessage(null);
    setError(null);
    try {
      const job = await api.deployGpu();
      jobRef.current = job.id;
      setSettings((s) => (s ? { ...s, gpu_status: "deploying", gpu_error: null } : s));
      startPoll();
    } catch (err) {
      fail(err);
    } finally {
      // «идёт деплой» показывает статус deploying, busy — только сам запрос
      setBusy(false);
    }
  };

  const wrongPrefix = tokenId.trim().length > 0 && !tokenId.trim().startsWith("ak-");
  const deploying = settings?.gpu_status === "deploying";

  return (
    <div className="settings-page">
      <h1>Настройки</h1>

      <section className="settings-section">
        <h2>Режим разметки</h2>
        <p className="muted">
          По умолчанию платформа размечает документы на CPU движком{" "}
          <code>rapidocr</code> — он работает сразу, без аккаунтов и ключей.
          GPU-режим — необязательное ускорение: платформа разворачивает
          GPU-рецепт <strong>в ваш собственный аккаунт Modal</strong>, вы платите
          Modal напрямую по своему тарифу, а платформа только запускает деплой и
          вызывает эндпоинт. Результат в обоих режимах одинаковый по структуре —
          построчные полигоны с текстом и confidence.
        </p>
      </section>

      <section className="settings-section">
        <h2>Токен Modal</h2>
        <p className="muted">
          Нужен API-токен: установите Modal CLI и выполните{" "}
          <code>modal token new</code> — он выведет пару{" "}
          <code>token_id</code> (<code>ak-...</code>) и <code>token_secret</code>{" "}
          (<code>as-...</code>). Токены из дашборда вида <code>wk-</code>/
          <code>ws-</code> — прокси-токены для вызовов, для деплоя они не годятся.
          Секрет хранится в зашифрованном виде и никогда не возвращается обратно
          в интерфейс.
        </p>

        {settings?.modal_configured && (
          <p>
            Подключён токен <code>{settings.modal_token_id_masked}</code>{" "}
            <button onClick={() => void disconnect()} disabled={busy}>
              Отключить
            </button>
          </p>
        )}

        <form className="settings-form" onSubmit={saveToken}>
          <label>
            token id
            <input
              value={tokenId}
              onChange={(e) => setTokenId(e.target.value)}
              placeholder="ak-..."
              autoComplete="off"
            />
          </label>
          <label>
            token secret
            <input
              type="password"
              value={tokenSecret}
              onChange={(e) => setTokenSecret(e.target.value)}
              placeholder="as-..."
              autoComplete="new-password"
            />
          </label>
          <button
            type="submit"
            disabled={busy || !tokenId.trim() || !tokenSecret.trim()}
          >
            {settings?.modal_configured ? "Заменить токен" : "Сохранить"}
          </button>
        </form>

        {wrongPrefix && (
          <p className="warning">
            token id обычно начинается с <code>ak-</code>. Если у вас{" "}
            <code>wk-</code> — это прокси-токен, для деплоя нужен вывод{" "}
            <code>modal token new</code>.
          </p>
        )}
      </section>

      <section className="settings-section">
        <h2>GPU-движок</h2>
        {settings ? <GpuStatusBlock settings={settings} /> : <p className="muted">...</p>}
        <button
          onClick={() => void deploy()}
          disabled={busy || !settings?.modal_configured || deploying}
        >
          {settings?.gpu_status === "ready"
            ? "Развернуть заново"
            : "Подключить GPU"}
        </button>
        {!settings?.modal_configured && (
          <span className="muted"> — сначала сохраните токен Modal</span>
        )}
      </section>

      {message && <p className="muted">{message}</p>}
      {error && <p className="error">{error}</p>}
    </div>
  );
}

export default SettingsPage;
