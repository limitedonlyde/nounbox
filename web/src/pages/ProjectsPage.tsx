import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, errorText, Project, TASK_TITLES, TaskType } from "../api";

function ProjectsPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [name, setName] = useState("");
  const [taskType, setTaskType] = useState<TaskType>("detection");
  const [error, setError] = useState<string | null>(null);

  const load = () =>
    api
      .listProjects()
      .then(setProjects)
      .catch((e) => setError(errorText(e)));

  useEffect(() => {
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const create = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    try {
      await api.createProject(name.trim(), taskType);
      setName("");
      setError(null);
      void load();
    } catch (err) {
      setError(errorText(err));
    }
  };

  return (
    <div>
      <div className="page-head">
        <div className="page-head-main">
          <h1>Projects</h1>
          <span className="muted">
            name the classes you want found, drop in photos, review the boxes
          </span>
        </div>
      </div>

      <form className="project-form card" onSubmit={create}>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="New project name"
        />
        <select
          value={taskType}
          onChange={(e) => setTaskType(e.target.value as TaskType)}
        >
          <option value="detection">{TASK_TITLES.detection}</option>
          <option value="ocr">{TASK_TITLES.ocr}</option>
        </select>
        <button type="submit">Create</button>
      </form>

      {error && <p className="error">{error}</p>}

      {projects.length === 0 ? (
        <div className="empty">
          <p className="empty-title">No projects yet</p>
          <p className="muted">
            Create one above — a detection project finds objects you name, an OCR
            project reads text.
          </p>
        </div>
      ) : (
        <ul className="project-list">
          {projects.map((p) => (
            <li key={p.id}>
              <Link to={`/projects/${p.id}`}>
                <strong>{p.name}</strong>
              </Link>{" "}
              <span className="muted">
                {TASK_TITLES[p.task_type] ?? p.task_type} ·{" "}
                {new Date(p.created_at).toLocaleString()}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default ProjectsPage;
