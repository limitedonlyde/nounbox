import { useState } from "react";
import { annotationsBlockingDelete, api, errorText, ProjectClass } from "../api";

/** Имя класса уходит в движок как текстовый запрос — только латиница. */
const ENGLISH_ONLY = /^[A-Za-z][A-Za-z0-9 _'/-]*$/;
const HEX_COLOR = /^#[0-9a-fA-F]{6}$/;

const normalize = (raw: string) => raw.trim().replace(/\s+/g, " ");

const annotationsWord = (n: number) =>
  n % 10 === 1 && n % 100 !== 11 ? "аннотации" : "аннотациях";

interface Props {
  projectId: string;
  classes: ProjectClass[];
  onChanged: () => Promise<void>;
}

function ClassesPanel({ projectId, classes, onChanged }: Props) {
  const [name, setName] = useState("");
  const [bulkOpen, setBulkOpen] = useState(false);
  const [bulkText, setBulkText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const openBulk = () => {
    setBulkText(classes.map((c) => c.name).join("\n"));
    setError(null);
    setBulkOpen(true);
  };

  const add = async (e: React.FormEvent) => {
    e.preventDefault();
    const value = normalize(name);
    if (!value) return;
    if (!ENGLISH_ONLY.test(value)) {
      setError(
        `Имя «${value}» не английское. Пишите латиницей: carpet, sofa, chandelier.`
      );
      return;
    }
    if (classes.some((c) => c.name.toLowerCase() === value.toLowerCase())) {
      setError(`Класс «${value}» уже есть.`);
      return;
    }
    setBusy(true);
    try {
      await api.createClass(projectId, value);
      setName("");
      setError(null);
      await onChanged();
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  };

  const recolor = async (cls: ProjectClass, color: string) => {
    setBusy(true);
    try {
      await api.updateClass(cls.id, { color });
      setError(null);
      await onChanged();
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  };

  const remove = async (cls: ProjectClass) => {
    if (!window.confirm(`Удалить класс «${cls.name}»?`)) return;
    setBusy(true);
    try {
      await api.deleteClass(cls.id);
    } catch (err) {
      const used = annotationsBlockingDelete(err);
      if (used === null) {
        setError(errorText(err));
        setBusy(false);
        return;
      }
      const where =
        used > 0
          ? `в ${used} ${annotationsWord(used)}`
          : "в существующих аннотациях";
      if (
        !window.confirm(
          `Класс «${cls.name}» используется ${where}. Удалить вместе с ними?`
        )
      ) {
        setBusy(false);
        return;
      }
      try {
        await api.deleteClass(cls.id, true);
      } catch (forced) {
        setError(errorText(forced));
        setBusy(false);
        return;
      }
    }
    setError(null);
    await onChanged();
    setBusy(false);
  };

  const applyBulk = async (e: React.FormEvent) => {
    e.preventDefault();
    const names: string[] = [];
    const seen = new Set<string>();
    for (const line of bulkText.split("\n")) {
      const value = normalize(line);
      if (!value || seen.has(value.toLowerCase())) continue;
      if (!ENGLISH_ONLY.test(value)) {
        setError(
          `Строка «${value}» не английская. Пишите латиницей: carpet, sofa, chandelier.`
        );
        return;
      }
      seen.add(value.toLowerCase());
      names.push(value);
    }
    if (names.length === 0) {
      setError("Список пуст — добавьте хотя бы один класс.");
      return;
    }
    const dropped = classes.filter((c) => !seen.has(c.name.toLowerCase()));
    if (
      dropped.length > 0 &&
      !window.confirm(
        `Список заменит текущие классы. Будут удалены: ${dropped
          .map((c) => c.name)
          .join(", ")}. Продолжить?`
      )
    ) {
      return;
    }
    setBusy(true);
    try {
      await api.replaceClasses(projectId, names);
      setError(null);
      setBulkOpen(false);
      await onChanged();
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="classes-panel">
      <div className="classes-head">
        <h2>
          Классы проекта <span className="muted">({classes.length})</span>
        </h2>
        {bulkOpen ? (
          <button type="button" onClick={() => setBulkOpen(false)}>
            Свернуть список
          </button>
        ) : (
          <button type="button" onClick={openBulk}>
            Ввести списком
          </button>
        )}
      </div>

      <p className="hint">
        Имя класса — <strong>только английскими словами</strong>: «carpet»,
        «sofa», «chandelier». Модель ищет объекты по этому тексту, «ковёр» даст
        пустой результат.
      </p>

      <form className="class-add" onSubmit={(e) => void add(e)}>
        <label className="class-add-label">
          Новый класс (English)
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="carpet"
            spellCheck={false}
            autoCapitalize="off"
          />
        </label>
        <button type="submit" disabled={busy || !name.trim()}>
          Добавить
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      {classes.length === 0 ? (
        <p className="muted">
          Классов пока нет — разметка не запустится, пока не добавите хотя бы
          один.
        </p>
      ) : (
        <ul className="class-list">
          {classes.map((cls, i) => (
            <li key={cls.id}>
              <input
                type="color"
                className="class-color"
                // неконтролируемый + коммит по завершению выбора: onChange у
                // input[type=color] стреляет на каждое движение по палитре
                defaultValue={HEX_COLOR.test(cls.color) ? cls.color : "#3b82f6"}
                key={`${cls.id}-${cls.color}`}
                onBlur={(e) => {
                  if (e.target.value !== cls.color) void recolor(cls, e.target.value);
                }}
                title="цвет рамки в Review"
              />
              <span className="class-name">{cls.name}</span>
              {i < 9 && <kbd title="горячая клавиша в Review">{i + 1}</kbd>}
              <button
                type="button"
                className="class-remove"
                onClick={() => void remove(cls)}
                disabled={busy}
                title="удалить класс"
              >
                ✕
              </button>
            </li>
          ))}
        </ul>
      )}

      {bulkOpen && (
        <form className="class-bulk" onSubmit={(e) => void applyBulk(e)}>
          <label className="class-add-label">
            Одно английское имя в строке — список заменит текущие классы
            <textarea
              rows={6}
              value={bulkText}
              onChange={(e) => setBulkText(e.target.value)}
              placeholder={"carpet\nsofa\nchandelier"}
              spellCheck={false}
            />
          </label>
          <div className="class-bulk-actions">
            <button type="submit" disabled={busy}>
              Сохранить список
            </button>
            <button type="button" onClick={() => setBulkOpen(false)}>
              Отмена
            </button>
          </div>
        </form>
      )}
    </section>
  );
}

export default ClassesPanel;
