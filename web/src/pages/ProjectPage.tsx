import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  api,
  DEFAULT_ACCEPT_THRESHOLD,
  DEFAULT_LABELER,
  DEFAULT_SCORE_THRESHOLD,
  errorText,
  EXPORT_FORMATS,
  ImageListItem,
  Labeler,
  labelerSupportsTask,
  Project,
  ProjectClass,
  TASK_TITLES,
} from "../api";
import ClassesPanel from "../components/ClassesPanel";

const CONFIG_PLACEHOLDER: Record<string, string> = {
  vlm: 'config JSON: {"base_url": "...", "model": "..."}',
  http: 'config JSON: {"endpoint": "https://.../predict"}',
  consensus: 'config JSON: {"engines": [{"name": "rapidocr"}, {"name": "vlm"}]}',
};

function Thumb({ imageId }: { imageId: string }) {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    api.getImageUrl(imageId).then((u) => alive && setUrl(u));
    return () => {
      alive = false;
    };
  }, [imageId]);
  return url ? <img src={url} alt="" /> : <div className="thumb-loading" />;
}

function ProjectPage() {
  const { projectId = "" } = useParams();
  const [project, setProject] = useState<Project | null>(null);
  const [classes, setClasses] = useState<ProjectClass[]>([]);
  const [images, setImages] = useState<ImageListItem[]>([]);
  const [scoreThreshold, setScoreThreshold] = useState(DEFAULT_SCORE_THRESHOLD);
  const [acceptThreshold, setAcceptThreshold] = useState(
    DEFAULT_ACCEPT_THRESHOLD.detection
  );
  const [exportFormat, setExportFormat] = useState("");
  const [labelers, setLabelers] = useState<Labeler[]>([]);
  const [engine, setEngine] = useState("");
  const [engineConfig, setEngineConfig] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const pollRef = useRef<number | null>(null);
  const defaultsApplied = useRef(false);

  const taskType = project?.task_type ?? "detection";

  const stopPoll = useCallback(() => {
    if (pollRef.current !== null) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  useEffect(() => stopPoll, [stopPoll]);

  const loadClasses = useCallback(async () => {
    setClasses(await api.listClasses(projectId));
  }, [projectId]);

  const load = useCallback(async () => {
    const [p, imgs, cls] = await Promise.all([
      api.getProject(projectId),
      api.listProjectImages(projectId),
      api.listClasses(projectId),
    ]);
    setProject(p);
    setImages(imgs);
    setClasses(cls);
    // fill the thresholds in once — after that the value belongs to the user
    if (!defaultsApplied.current) {
      defaultsApplied.current = true;
      setAcceptThreshold(DEFAULT_ACCEPT_THRESHOLD[p.task_type]);
    }
  }, [projectId]);

  useEffect(() => {
    load().catch((err) => setMessage(`Failed to load: ${errorText(err)}`));
  }, [load]);

  const watchJob = useCallback(
    (jobId: string) => {
      setBusy(true);
      stopPoll();
      // a single network blip must not kill the poll: count misses in a row
      let misses = 0;
      pollRef.current = window.setInterval(async () => {
        try {
          const j = await api.getJob(jobId);
          misses = 0;
          if (j.status !== "done" && j.status !== "failed") return;
          stopPoll();
          setBusy(false);
          setMessage(
            j.status === "done"
              ? Number(j.result.skipped_already_labeled ?? 0) > 0 &&
                Number(j.result.annotations_created ?? 0) === 0
                ? "Every image is already labeled by this engine. If you " +
                  "changed the class list, use “Relabel all”."
                : `Done. Annotations created: ${String(j.result.annotations_created ?? 0)}`
              : `Error: ${String(j.result.error ?? "unknown")}`
          );
          void load();
        } catch (err) {
          misses += 1;
          if (misses < 3) return;
          stopPoll();
          setBusy(false);
          setMessage(`Error: ${errorText(err)}`);
        }
      }, 3000);
    },
    [stopPoll, load]
  );

  // pick up an unfinished job when the page opens: the tab may have been
  // reloaded, put to sleep or lost the network — without this the interface
  // stays stuck on "Autolabel started..." forever, while the work runs and ends
  useEffect(() => {
    let alive = true;
    api
      .getActiveJob(projectId)
      .then((job) => {
        if (!alive || !job) return;
        setMessage("Autolabel is still running...");
        watchJob(job.id);
      })
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, [projectId, watchJob]);

  useEffect(() => {
    let alive = true;
    api
      .listLabelers()
      .then((list) => alive && setLabelers(list))
      .catch(
        (err) =>
          alive &&
          setMessage(`Could not load the engine list: ${errorText(err)}`)
      );
    return () => {
      alive = false;
    };
  }, []);

  // engines that fit the project's task type
  const taskLabelers = useMemo(
    () => labelers.filter((l) => labelerSupportsTask(l, taskType)),
    [labelers, taskType]
  );

  // default engine: owlv2 for detection, rapidocr for ocr
  useEffect(() => {
    if (taskLabelers.length === 0) return;
    setEngine((current) => {
      const picked = taskLabelers.find((l) => l.name === current);
      if (picked?.available) return current;
      const preferred = DEFAULT_LABELER[taskType];
      const fallback =
        taskLabelers.find((l) => l.name === preferred && l.available) ??
        taskLabelers.find((l) => l.available) ??
        taskLabelers[0];
      return fallback.name;
    });
  }, [taskLabelers, taskType]);

  // export formats depend on the task type
  const exportFormats = EXPORT_FORMATS[taskType];
  useEffect(() => {
    setExportFormat((current) =>
      exportFormats.some((f) => f.value === current)
        ? current
        : exportFormats[0].value
    );
  }, [exportFormats]);

  const selectedLabeler = taskLabelers.find((l) => l.name === engine) ?? null;
  const needsConfig = selectedLabeler?.requires === "config";
  const engineBlocked = taskLabelers.length > 0 && !selectedLabeler?.available;
  const isDetection = taskType === "detection";
  const classesMissing = isDetection && classes.length === 0;

  const upload = async (files: FileList | null) => {
    if (!files?.length) return;
    setBusy(true);
    setMessage("Uploading and ingesting...");
    for (const file of Array.from(files)) {
      await api.uploadDocument(projectId, file);
    }
    // ingest runs in the background — refresh the list in a couple of seconds
    setTimeout(() => void load(), 3000);
    setBusy(false);
    setMessage(`Files uploaded: ${files.length}. Ingest runs in the background.`);
  };

  const autolabel = async (rerun = false) => {
    setBusy(true);
    setMessage("Autolabel started...");
    let config: Record<string, unknown> = {};
    if (needsConfig) {
      try {
        config = engineConfig.trim() ? JSON.parse(engineConfig) : {};
      } catch {
        setBusy(false);
        setMessage("Error: config is not valid JSON");
        return;
      }
    }
    // the worker fills in the class list; from here we only set the display
    // threshold (if the user set it explicitly in the JSON config, leave it)
    if (isDetection && config.score_threshold === undefined) {
      config.score_threshold = scoreThreshold;
    }
    let job;
    try {
      job = await api.startAutolabel(projectId, engine, config, rerun);
    } catch (err) {
      setBusy(false);
      setMessage(`Error: ${errorText(err)}`);
      return;
    }
    watchJob(job.id);
  };

  const bulkAccept = async () => {
    const { accepted } = await api.bulkAccept(projectId, acceptThreshold);
    setMessage(`Annotations accepted: ${accepted} (confidence ≥ ${acceptThreshold})`);
    void load();
  };

  const totalPending = images.reduce((s, i) => s + i.pending_annotations, 0);
  const totalAnnotations = images.reduce((s, i) => s + i.total_annotations, 0);

  const autolabelHint = classesMissing
    ? "Add project classes before you start labeling"
    : selectedLabeler?.reason ?? undefined;

  return (
    <div>
      <div className="page-head">
        <div className="page-head-main">
          <Link to="/" className="crumb">
            ← Projects
          </Link>
          <h1>{project?.name ?? "..."}</h1>
          <span className="muted">{TASK_TITLES[taskType]}</span>
        </div>
        <div className="page-head-actions">
          <button onClick={() => fileInput.current?.click()} disabled={busy}>
            Upload files
          </button>
        </div>
      </div>

      <div className="stat-row">
        <div className="stat">
          <div className="stat-value">{images.length}</div>
          <div className="stat-label">Images</div>
        </div>
        <div className="stat">
          <div className="stat-value">{totalAnnotations}</div>
          <div className="stat-label">Annotations</div>
        </div>
        <div className={totalPending > 0 ? "stat stat-accent" : "stat"}>
          <div className="stat-value">{totalPending}</div>
          <div className="stat-label">Pending review</div>
        </div>
      </div>

      {isDetection && (
        <ClassesPanel
          projectId={projectId}
          classes={classes}
          onChanged={loadClasses}
        />
      )}

      <div className="card">
        <div className="card-head">
          <h2>Labeling</h2>
          <span className="muted">an engine draws the first pass; you review it</span>
        </div>
        <div className="card-body toolbar">
        <input
          ref={fileInput}
          type="file"
          multiple
          hidden
          onChange={(e) => void upload(e.target.files)}
        />
        <select
          value={engine}
          onChange={(e) => setEngine(e.target.value)}
          disabled={taskLabelers.length === 0}
          title={selectedLabeler?.reason ?? undefined}
        >
          {taskLabelers.map((l) => (
            <option key={l.name} value={l.name} disabled={!l.available}>
              {l.title}
              {l.available ? "" : ` — ${l.reason ?? "unavailable"}`}
            </option>
          ))}
        </select>
        {needsConfig && (
          <input
            className="config-input"
            value={engineConfig}
            onChange={(e) => setEngineConfig(e.target.value)}
            placeholder={CONFIG_PLACEHOLDER[engine] ?? "config JSON: {}"}
          />
        )}
        {isDetection && (
          <label>
            engine threshold{" "}
            <input
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={scoreThreshold}
              onChange={(e) => setScoreThreshold(Number(e.target.value))}
              title="boxes below this score are not shown"
            />
          </label>
        )}
        <button
          className="primary"
          onClick={() => void autolabel()}
          disabled={
            busy ||
            images.length === 0 ||
            // without an engine name the server would run ALL installed ones at once
            !engine ||
            engineBlocked ||
            classesMissing
          }
          title={autolabelHint}
        >
          Autolabel
        </button>
        <button
          onClick={() => void autolabel(true)}
          disabled={
            busy ||
            images.length === 0 ||
            !engine ||
            engineBlocked ||
            classesMissing ||
            totalAnnotations === 0
          }
          title="Run again on every image: unreviewed engine boxes are replaced,
the ones you accepted or fixed stay"
        >
          Relabel all
        </button>
        {classesMissing ? (
          <span className="warning">
            Add project classes before you start labeling
          </span>
        ) : (
          engineBlocked && (
            <span className="muted">
              {selectedLabeler?.reason ?? "engine unavailable"} —{" "}
              <Link to="/settings">settings</Link>
            </span>
          )
        )}
        <span className="toolbar-group">
          <label>
            accept ≥{" "}
            <input
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={acceptThreshold}
              onChange={(e) => setAcceptThreshold(Number(e.target.value))}
            />
          </label>
          <button onClick={() => void bulkAccept()} disabled={totalPending === 0}>
            Accept all above threshold
          </button>
        </span>
        <span className="toolbar-group">
          <select
            value={exportFormat}
            onChange={(e) => setExportFormat(e.target.value)}
          >
            {exportFormats.map((f) => (
              <option key={f.value} value={f.value}>
                {f.title}
              </option>
            ))}
          </select>
          <a
            className="export-link"
            href={`/api/v1/projects/${projectId}/export?format=${exportFormat}`}
          >
            Export ⤓
          </a>
        </span>
      </div>

      </div>

      {message && <p className="muted">{message}</p>}

      <div className="card">
        <div className="card-head">
          <h2>Images</h2>
          <span className="muted">
            {images.length === 0
              ? "nothing uploaded yet"
              : `${images.length} in this project`}
          </span>
        </div>
        <div className="card-body image-grid">
        {images.map((img) => (
          <Link
            key={img.id}
            to={`/projects/${projectId}/review/${img.id}`}
            className="image-card"
          >
            <Thumb imageId={img.id} />
            <div className="image-card-info">
              page {img.page_index + 1} · {img.total_annotations} boxes
              {img.pending_annotations > 0 && (
                <span className="badge-pending"> {img.pending_annotations} ⏳</span>
              )}
            </div>
          </Link>
        ))}
        </div>
      </div>
    </div>
  );
}

export default ProjectPage;
