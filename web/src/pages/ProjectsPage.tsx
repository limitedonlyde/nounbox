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
      <h1>Projects</h1>

      <form className="project-form" onSubmit={create}>
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
        <p className="muted">No projects yet — create the first one.</p>
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
