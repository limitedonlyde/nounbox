import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import {
  Annotation,
  AnnotationStatus,
  api,
  errorText,
  Geometry,
  ImageListItem,
  ImageMeta,
  Project,
  ProjectClass,
} from "../api";

const STATUS_COLORS: Record<AnnotationStatus, string> = {
  pending: "#f0a020",
  accepted: "#2fa84f",
  edited: "#2f7df6",
  rejected: "#e05252",
};

/** Класс аннотации удалён из проекта — рисуем серым, но не прячем. */
const UNKNOWN_CLASS_COLOR = "#8a8a8a";

type StatusFilter = "all" | AnnotationStatus;

const geometryAnchor = (g: Geometry): { x: number; y: number } =>
  g.type === "bbox"
    ? { x: g.x, y: g.y }
    : {
        x: Math.min(...g.points.map((p) => p[0])),
        y: Math.min(...g.points.map((p) => p[1])),
      };

function ReviewPage() {
  const { projectId = "", imageId = "" } = useParams();
  const navigate = useNavigate();

  const [project, setProject] = useState<Project | null>(null);
  const [classes, setClasses] = useState<ProjectClass[]>([]);
  const [image, setImage] = useState<ImageMeta | null>(null);
  const [url, setUrl] = useState("");
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [images, setImages] = useState<ImageListItem[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [activeClass, setActiveClass] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [minConf, setMinConf] = useState(0);
  const [drawMode, setDrawMode] = useState(false);
  const [draftRect, setDraftRect] = useState<{ x: number; y: number; w: number; h: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const svgRef = useRef<SVGSVGElement>(null);
  const textRef = useRef<HTMLTextAreaElement>(null);
  const drawStart = useRef<{ x: number; y: number } | null>(null);

  // --- загрузка проекта и классов ---
  useEffect(() => {
    let alive = true;
    Promise.all([
      api.getProject(projectId),
      api.listClasses(projectId),
      api.listProjectImages(projectId),
    ])
      .then(([p, cls, imgs]) => {
        if (!alive) return;
        setProject(p);
        setClasses(cls);
        setImages(imgs);
        setActiveClass((current) =>
          current && cls.some((c) => c.name === current)
            ? current
            : cls[0]?.name ?? null
        );
      })
      .catch((err) => alive && setError(errorText(err)));
    return () => {
      alive = false;
    };
  }, [projectId]);

  // --- загрузка изображения ---
  useEffect(() => {
    let alive = true;
    setSelectedId(null);
    Promise.all([
      api.getImage(imageId),
      api.getImageUrl(imageId),
      api.listAnnotations(imageId),
    ])
      .then(([meta, u, anns]) => {
        if (!alive) return;
        setImage(meta);
        setUrl(u);
        setAnnotations(anns);
      })
      .catch((err) => alive && setError(errorText(err)));
    return () => {
      alive = false;
    };
  }, [imageId]);

  const isOcr = project?.task_type === "ocr";

  const classByName = useMemo(
    () => new Map(classes.map((c) => [c.name, c])),
    [classes]
  );

  const colorOf = useCallback(
    (a: Annotation) =>
      isOcr
        ? STATUS_COLORS[a.status]
        : classByName.get(a.label)?.color ?? UNKNOWN_CLASS_COLOR,
    [isOcr, classByName]
  );

  // --- фильтр: сначала низкоуверенные (очередь ревью) ---
  const filtered = useMemo(
    () =>
      annotations
        .filter(
          (a) =>
            (statusFilter === "all" || a.status === statusFilter) &&
            a.confidence >= minConf
        )
        .sort((a, b) => a.confidence - b.confidence),
    [annotations, statusFilter, minConf]
  );

  const selected = annotations.find((a) => a.id === selectedId) ?? null;

  const countByClass = useMemo(() => {
    const counts = new Map<string, number>();
    for (const a of annotations) counts.set(a.label, (counts.get(a.label) ?? 0) + 1);
    return counts;
  }, [annotations]);

  useEffect(() => {
    setDraft(selected?.text ?? "");
  }, [selectedId]); // eslint-disable-line react-hooks/exhaustive-deps

  // --- действия ---
  const patch = useCallback(
    async (id: string, body: Parameters<typeof api.updateAnnotation>[1]) => {
      const updated = await api.updateAnnotation(id, body);
      setAnnotations((as) => as.map((a) => (a.id === id ? updated : a)));
      return updated;
    },
    []
  );

  const setStatusAndAdvance = useCallback(
    async (id: string, status: AnnotationStatus) => {
      await patch(id, { status });
      setSelectedId((current) => {
        if (current !== id) return current;
        const next = filtered.find((a) => a.id !== id && a.status === "pending");
        return next?.id ?? null;
      });
    },
    [filtered, patch]
  );

  const remove = useCallback(async (id: string) => {
    try {
      await api.deleteAnnotation(id);
    } catch (err) {
      setError(errorText(err));
      return;
    }
    setError(null);
    setAnnotations((as) => as.filter((a) => a.id !== id));
    setSelectedId((current) => (current === id ? null : current));
  }, []);

  /** Назначить класс аннотации; он же становится классом для новых боксов. */
  const assignClass = useCallback(
    async (id: string, className: string) => {
      setActiveClass(className);
      try {
        await patch(id, { label: className });
        setError(null);
      } catch (err) {
        setError(errorText(err));
      }
    },
    [patch]
  );

  const pickClass = useCallback(
    (className: string) => {
      setActiveClass(className);
      if (selectedId) void assignClass(selectedId, className);
    },
    [selectedId, assignClass]
  );

  const saveDraft = useCallback(async () => {
    if (!selected || draft === (selected.text ?? "")) return;
    await patch(selected.id, { text: draft });
  }, [selected, draft, patch]);

  const navigateSibling = useCallback(
    (delta: number) => {
      const idx = images.findIndex((i) => i.id === imageId);
      const target = images[idx + delta];
      if (target) navigate(`/projects/${projectId}/review/${target.id}`);
    },
    [images, imageId, navigate, projectId]
  );

  const canDraw = isOcr || activeClass !== null;

  // --- горячие клавиши ---
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === "TEXTAREA" || tag === "INPUT" || tag === "SELECT") return;

      const idx = filtered.findIndex((a) => a.id === selectedId);
      switch (e.key) {
        case "ArrowDown":
        case "ArrowUp": {
          e.preventDefault();
          if (filtered.length === 0) return;
          const next =
            e.key === "ArrowDown"
              ? filtered[Math.min(idx + 1, filtered.length - 1)] ?? filtered[0]
              : filtered[Math.max(idx - 1, 0)] ?? filtered[filtered.length - 1];
          setSelectedId(next.id);
          return;
        }
        case "ArrowRight":
          navigateSibling(1);
          return;
        case "ArrowLeft":
          navigateSibling(-1);
          return;
        case "Delete":
        case "Backspace":
          if (selectedId) void remove(selectedId);
          return;
      }

      // цифры/буквы — по e.code: работают при CapsLock/Shift и на любой раскладке
      if (e.metaKey || e.ctrlKey || e.altKey) return;

      // 1-9 — класс выделенной аннотации (и класс для следующих боксов);
      // code — чтобы работало на любой раскладке, key — запасной вариант
      const digit =
        /^(?:Digit|Numpad)([1-9])$/.exec(e.code) ?? /^([1-9])$/.exec(e.key);
      if (digit) {
        if (isOcr) return;
        const cls = classes[Number(digit[1]) - 1];
        if (!cls) return;
        e.preventDefault();
        pickClass(cls.name);
        return;
      }

      switch (e.code) {
        case "KeyA":
          if (selectedId) void setStatusAndAdvance(selectedId, "accepted");
          break;
        case "KeyR":
          if (selectedId) void setStatusAndAdvance(selectedId, "rejected");
          break;
        case "KeyE":
          if (isOcr && selectedId) {
            e.preventDefault();
            textRef.current?.focus();
          }
          break;
        case "KeyD":
          if (canDraw) setDrawMode((v) => !v);
          break;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [
    filtered,
    selectedId,
    navigateSibling,
    setStatusAndAdvance,
    remove,
    classes,
    isOcr,
    pickClass,
    canDraw,
  ]);

  // --- рисование нового бокса ---
  const toImageCoords = (e: React.PointerEvent) => {
    const rect = svgRef.current!.getBoundingClientRect();
    if (!image) return { x: 0, y: 0 };
    return {
      x: ((e.clientX - rect.left) / rect.width) * image.width,
      y: ((e.clientY - rect.top) / rect.height) * image.height,
    };
  };

  const onPointerDown = (e: React.PointerEvent) => {
    if (!drawMode) return;
    drawStart.current = toImageCoords(e);
    setDraftRect({ ...drawStart.current, w: 0, h: 0 });
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (!drawMode || !drawStart.current) return;
    const p = toImageCoords(e);
    setDraftRect({
      x: Math.min(drawStart.current.x, p.x),
      y: Math.min(drawStart.current.y, p.y),
      w: Math.abs(p.x - drawStart.current.x),
      h: Math.abs(p.y - drawStart.current.y),
    });
  };

  const onPointerUp = async () => {
    if (!drawMode || !draftRect) return;
    drawStart.current = null;
    const rect = draftRect;
    setDraftRect(null);
    setDrawMode(false);
    if (rect.w < 8 || rect.h < 8) return; // случайный клик
    const geometry: Geometry = {
      type: "bbox",
      x: Math.round(rect.x),
      y: Math.round(rect.y),
      width: Math.round(rect.w),
      height: Math.round(rect.h),
    };
    let created: Annotation;
    try {
      created = await api.createAnnotation(
        imageId,
        geometry,
        isOcr ? { text: "" } : { label: activeClass ?? "" }
      );
    } catch (err) {
      setError(errorText(err));
      return;
    }
    setError(null);
    setAnnotations((as) => [...as, created]);
    setSelectedId(created.id);
    if (isOcr) setTimeout(() => textRef.current?.focus(), 0);
  };

  if (!image) return <p className="muted">{error ?? "Загрузка..."}</p>;

  const imageIndex = images.findIndex((i) => i.id === imageId);
  const done = annotations.filter(
    (a) => a.status === "accepted" || a.status === "edited"
  ).length;
  // подписи в SVG живут в координатах картинки — размер считаем от неё
  const labelSize = Math.max(10, Math.round(Math.max(image.width, image.height) / 55));
  const activeColor = activeClass
    ? classByName.get(activeClass)?.color ?? UNKNOWN_CLASS_COLOR
    : "#2f7df6";

  return (
    <div>
      <div className="toolbar">
        <Link to={`/projects/${projectId}`}>← к списку</Link>
        <span className="muted">
          изображение {imageIndex + 1} / {images.length} · проверено {done}/
          {annotations.length}
        </span>
        <button
          className={drawMode ? "active" : ""}
          onClick={() => setDrawMode((v) => !v)}
          disabled={!canDraw}
          title={
            canDraw
              ? "D"
              : "Сначала добавьте классы проекта — новому боксу нужен класс"
          }
        >
          + бокс
          {!isOcr && activeClass ? `: ${activeClass}` : ""}
        </button>
        <span className="toolbar-group">
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value as StatusFilter)}
          >
            <option value="all">все статусы</option>
            <option value="pending">pending</option>
            <option value="accepted">accepted</option>
            <option value="edited">edited</option>
            <option value="rejected">rejected</option>
          </select>
          <label>
            conf ≥{" "}
            <input
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={minConf}
              onChange={(e) => setMinConf(Number(e.target.value))}
            />
          </label>
        </span>
      </div>

      {error && <p className="error">{error}</p>}

      <div className="review-layout">
        <div className="review-canvas">
          {url && <img src={url} alt="" draggable={false} />}
          <svg
            ref={svgRef}
            viewBox={`0 0 ${image.width} ${image.height}`}
            className={drawMode ? "drawing" : ""}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={() => void onPointerUp()}
          >
            {filtered.map((a) => {
              const color = colorOf(a);
              const isSelected = a.id === selectedId;
              const common = {
                stroke: color,
                strokeWidth: isSelected ? 3 : 1.5,
                strokeDasharray:
                  a.status === "pending"
                    ? "6 4"
                    : a.status === "rejected"
                      ? "2 4"
                      : undefined,
                vectorEffect: "non-scaling-stroke" as const,
                fill: isSelected ? `${color}33` : "transparent",
                opacity: a.status === "rejected" ? 0.45 : 1,
                onClick: (e: React.MouseEvent) => {
                  if (drawMode) return;
                  e.stopPropagation();
                  setSelectedId(a.id);
                },
                style: { cursor: drawMode ? "crosshair" : "pointer" },
              };
              const anchor = geometryAnchor(a.geometry);
              return (
                <g key={a.id}>
                  {a.geometry.type === "polygon" ? (
                    <polygon
                      points={a.geometry.points.map((p) => p.join(",")).join(" ")}
                      {...common}
                    />
                  ) : (
                    <rect
                      x={a.geometry.x}
                      y={a.geometry.y}
                      width={a.geometry.width}
                      height={a.geometry.height}
                      {...common}
                    />
                  )}
                  {!isOcr && (
                    <text
                      x={anchor.x}
                      y={
                        anchor.y > labelSize
                          ? anchor.y - labelSize * 0.3
                          : anchor.y + labelSize
                      }
                      className="box-label"
                      fontSize={labelSize}
                      fill={color}
                      opacity={a.status === "rejected" ? 0.45 : 1}
                    >
                      {a.label}
                    </text>
                  )}
                </g>
              );
            })}
            {draftRect && (
              <rect
                x={draftRect.x}
                y={draftRect.y}
                width={draftRect.w}
                height={draftRect.h}
                className="draft-rect"
                stroke={activeColor}
              />
            )}
          </svg>
        </div>

        <aside className="review-panel">
          {!isOcr && (
            <div className="class-legend">
              <div className="class-legend-head">
                Классы <span className="muted">(1-9 — назначить)</span>
              </div>
              {classes.length === 0 ? (
                <p className="muted">
                  Классов нет —{" "}
                  <Link to={`/projects/${projectId}`}>добавьте их в проекте</Link>.
                </p>
              ) : (
                <ul>
                  {classes.map((c, i) => (
                    <li
                      key={c.id}
                      className={c.name === activeClass ? "active" : ""}
                      onClick={() => pickClass(c.name)}
                      title="назначить выделенной аннотации"
                    >
                      <span className="class-dot" style={{ background: c.color }} />
                      <span className="class-name">{c.name}</span>
                      {i < 9 && <kbd>{i + 1}</kbd>}
                      <span className="ann-conf">{countByClass.get(c.name) ?? 0}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}

          {selected ? (
            <div className="editor">
              <div className="editor-meta">
                <span style={{ color: STATUS_COLORS[selected.status] }}>●</span>{" "}
                {selected.status} · {Math.round(selected.confidence * 100)}% ·{" "}
                {selected.source.name}
              </div>

              {isOcr ? (
                <textarea
                  ref={textRef}
                  value={draft}
                  rows={3}
                  onChange={(e) => setDraft(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      void saveDraft().then(() => textRef.current?.blur());
                    }
                    if (e.key === "Escape") textRef.current?.blur();
                  }}
                />
              ) : (
                <label className="editor-field">
                  класс
                  <select
                    value={selected.label}
                    disabled={classes.length === 0}
                    onChange={(e) => void assignClass(selected.id, e.target.value)}
                  >
                    {!classByName.has(selected.label) && (
                      <option value={selected.label}>
                        {selected.label || "—"} — нет в проекте
                      </option>
                    )}
                    {classes.map((c, i) => (
                      <option key={c.id} value={c.name}>
                        {i < 9 ? `${i + 1}. ` : ""}
                        {c.name}
                      </option>
                    ))}
                  </select>
                </label>
              )}

              <div className="editor-actions">
                {isOcr && <button onClick={() => void saveDraft()}>Сохранить</button>}
                <button onClick={() => void setStatusAndAdvance(selected.id, "accepted")}>
                  ✓ <kbd>A</kbd>
                </button>
                <button onClick={() => void setStatusAndAdvance(selected.id, "rejected")}>
                  ✗ <kbd>R</kbd>
                </button>
                <button onClick={() => void remove(selected.id)}>🗑</button>
              </div>
            </div>
          ) : (
            <p className="muted">
              Кликните аннотацию. Клавиши: ↑↓ — выбор, ←→ — страницы, A —
              принять, R — отклонить, {isOcr ? "E — текст, " : "1-9 — класс, "}D
              — новый бокс, Del — удалить.
            </p>
          )}

          <ul className="ann-list">
            {filtered.map((a) => (
              <li
                key={a.id}
                className={a.id === selectedId ? "selected" : ""}
                onClick={() => setSelectedId(a.id)}
                ref={a.id === selectedId ? (el) => el?.scrollIntoView({ block: "nearest" }) : undefined}
              >
                <span style={{ color: STATUS_COLORS[a.status] }}>●</span>{" "}
                <span className="ann-conf">{Math.round(a.confidence * 100)}%</span>{" "}
                {isOcr ? (
                  <span className="ann-text">{a.text || "—"}</span>
                ) : (
                  <span className="ann-text" style={{ color: colorOf(a) }}>
                    {a.label || "—"}
                  </span>
                )}
              </li>
            ))}
          </ul>
        </aside>
      </div>
    </div>
  );
}

export default ReviewPage;
