import { useState } from "react";
import { annotationsBlockingDelete, api, errorText, ProjectClass } from "../api";

/** The class name goes to the engine as a text query — Latin letters only. */
const ENGLISH_ONLY = /^[A-Za-z][A-Za-z0-9 _'/-]*$/;
const HEX_COLOR = /^#[0-9a-fA-F]{6}$/;

const normalize = (raw: string) => raw.trim().replace(/\s+/g, " ");

const annotationsWord = (n: number) => (n === 1 ? "annotation" : "annotations");

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
        `The name “${value}” is not English. Use Latin letters: carpet, sofa, chandelier.`
      );
      return;
    }
    if (classes.some((c) => c.name.toLowerCase() === value.toLowerCase())) {
      setError(`Class “${value}” already exists.`);
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
    if (!window.confirm(`Delete class “${cls.name}”?`)) return;
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
          ? `in ${used} ${annotationsWord(used)}`
          : "in existing annotations";
      if (
        !window.confirm(`Class “${cls.name}” is used ${where}. Delete them too?`)
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
          `The line “${value}” is not English. Use Latin letters: carpet, sofa, chandelier.`
        );
        return;
      }
      seen.add(value.toLowerCase());
      names.push(value);
    }
    if (names.length === 0) {
      setError("The list is empty — add at least one class.");
      return;
    }
    const dropped = classes.filter((c) => !seen.has(c.name.toLowerCase()));
    if (
      dropped.length > 0 &&
      !window.confirm(
        `This list replaces the current classes. These will be deleted: ${dropped
          .map((c) => c.name)
          .join(", ")}. Continue?`
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
          Project classes <span className="muted">({classes.length})</span>
        </h2>
        {bulkOpen ? (
          <button type="button" onClick={() => setBulkOpen(false)}>
            Collapse list
          </button>
        ) : (
          <button type="button" onClick={openBulk}>
            Enter as list
          </button>
        )}
      </div>

      <p className="hint">
        A class name must be <strong>plain English words</strong>: “carpet”,
        “sofa”, “chandelier”. The model looks for objects by this text, so a name
        in another language or script returns nothing.
      </p>

      <form className="class-add" onSubmit={(e) => void add(e)}>
        <label className="class-add-label">
          New class (English)
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="carpet"
            spellCheck={false}
            autoCapitalize="off"
          />
        </label>
        <button type="submit" disabled={busy || !name.trim()}>
          Add
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      {classes.length === 0 ? (
        <p className="muted">
          No classes yet — labeling will not start until you add at least one.
        </p>
      ) : (
        <ul className="class-list">
          {classes.map((cls, i) => (
            <li key={cls.id}>
              <input
                type="color"
                className="class-color"
                // uncontrolled + commit on blur, once the user is done picking: onChange on
                // input[type=color] fires on every move across the palette
                defaultValue={HEX_COLOR.test(cls.color) ? cls.color : "#3b82f6"}
                key={`${cls.id}-${cls.color}`}
                onBlur={(e) => {
                  if (e.target.value !== cls.color) void recolor(cls, e.target.value);
                }}
                title="box color in Review"
              />
              <span className="class-name">{cls.name}</span>
              {i < 9 && <kbd title="hotkey in Review">{i + 1}</kbd>}
              <button
                type="button"
                className="class-remove"
                onClick={() => void remove(cls)}
                disabled={busy}
                title="delete class"
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
            One English name per line — this list replaces the current classes
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
              Save list
            </button>
            <button type="button" onClick={() => setBulkOpen(false)}>
              Cancel
            </button>
          </div>
        </form>
      )}
    </section>
  );
}

export default ClassesPanel;
