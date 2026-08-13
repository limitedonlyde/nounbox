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

/** The annotation's class was deleted from the project — draw it gray, do not hide. */
const UNKNOWN_CLASS_COLOR = "#8a8a8a";

type StatusFilter = "all" | AnnotationStatus;

/** The visible area in image coordinates (zooming in = shrinking w/h). */
interface Viewport {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Handles: the corners and the midpoints of the sides. */
type Handle = "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w";

const HANDLES: Handle[] = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];

const HANDLE_CURSOR: Record<Handle, string> = {
  nw: "nwse-resize",
  se: "nwse-resize",
  ne: "nesw-resize",
  sw: "nesw-resize",
  n: "ns-resize",
  s: "ns-resize",
  e: "ew-resize",
  w: "ew-resize",
};

interface Box {
  x: number;
  y: number;
  width: number;
  height: number;
}

type DragState =
  | { mode: "resize"; id: string; handle: Handle; before: Box; box: Box }
  | { mode: "move"; id: string; before: Box; box: Box; grab: { x: number; y: number } }
  | { mode: "pan"; startPointer: { x: number; y: number }; startView: Viewport };

interface GeometryEdit {
  id: string;
  before: Geometry;
  after: Geometry;
}

/** Minimum side of a box, in image pixels. */
const MIN_BOX = 4;
const ZOOM_MAX = 20;

const asBox = (g: Geometry): Box | null =>
  g.type === "bbox" ? { x: g.x, y: g.y, width: g.width, height: g.height } : null;

const boxGeometry = (b: Box): Geometry => ({
  type: "bbox",
  x: Math.round(b.x),
  y: Math.round(b.y),
  width: Math.round(b.width),
  height: Math.round(b.height),
});

/** Dragging a handle: recompute the edges, clamped to the frame and min size. */
function resizeBox(
  before: Box,
  handle: Handle,
  p: { x: number; y: number },
  imgW: number,
  imgH: number
): Box {
  let { x, y } = before;
  let right = before.x + before.width;
  let bottom = before.y + before.height;
  const px = Math.max(0, Math.min(p.x, imgW));
  const py = Math.max(0, Math.min(p.y, imgH));

  if (handle.includes("w")) x = Math.min(px, right - MIN_BOX);
  if (handle.includes("e")) right = Math.max(px, x + MIN_BOX);
  if (handle.includes("n")) y = Math.min(py, bottom - MIN_BOX);
  if (handle.includes("s")) bottom = Math.max(py, y + MIN_BOX);

  return { x, y, width: right - x, height: bottom - y };
}

/** Moving the whole box — we do not let it run outside the frame. */
function moveBox(before: Box, dx: number, dy: number, imgW: number, imgH: number): Box {
  return {
    ...before,
    x: Math.max(0, Math.min(before.x + dx, imgW - before.width)),
    y: Math.max(0, Math.min(before.y + dy, imgH - before.height)),
  };
}

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

  // zoom/pan: we keep the visible area in image coordinates
  const [view, setView] = useState<Viewport | null>(null);
  // the drag in progress: a resize, a box move, or a pan
  const [drag, setDrag] = useState<DragState | null>(null);
  // undo stack: geometry before/after, so that Cmd+Z puts the box back
  const undoStack = useRef<GeometryEdit[]>([]);
  const redoStack = useRef<GeometryEdit[]>([]);
  const [undoDepth, setUndoDepth] = useState(0);

  const svgRef = useRef<SVGSVGElement>(null);
  const textRef = useRef<HTMLTextAreaElement>(null);
  const drawStart = useRef<{ x: number; y: number } | null>(null);
  const spaceHeld = useRef(false);

  // --- loading the project and its classes ---
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

  // --- loading the image ---
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

  // --- filter: least confident first (the review queue) ---
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

  // --- actions ---
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

  /** Assign a class to the annotation; it also becomes the class for new boxes. */
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

  const zoomed = !!(image && view && view.w < image.width - 0.5);

  const resetView = useCallback(() => {
    if (image) setView({ x: 0, y: 0, w: image.width, h: image.height });
  }, [image]);

  // switching images resets the zoom and the undo stack — they are about the old frame
  useEffect(() => {
    if (image) setView({ x: 0, y: 0, w: image.width, h: image.height });
    undoStack.current = [];
    redoStack.current = [];
    setUndoDepth(0);
  }, [image]);

  const undo = useCallback(async () => {
    const last = undoStack.current.pop();
    setUndoDepth(undoStack.current.length);
    if (!last) return;
    redoStack.current.push(last);
    try {
      await patch(last.id, { geometry: last.before });
    } catch (err) {
      setError(errorText(err));
    }
  }, [patch]);

  const redo = useCallback(async () => {
    const next = redoStack.current.pop();
    if (!next) return;
    undoStack.current.push(next);
    setUndoDepth(undoStack.current.length);
    try {
      await patch(next.id, { geometry: next.after });
    } catch (err) {
      setError(errorText(err));
    }
  }, [patch]);

  /** The image is empty and reviewed: the only way to get it into the dataset
   * as a background example. */
  const markEmpty = useCallback(async () => {
    if (!image) return;
    try {
      const updated = await api.markImageReviewed(image.id, !image.reviewed);
      setImage(updated);
      setError(null);
    } catch (err) {
      setError(errorText(err));
    }
  }, [image]);

  const canDraw = isOcr || activeClass !== null;

  // --- hotkeys ---
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
        case " ":
          // space held down — the cursor turns into a "hand" for panning
          spaceHeld.current = true;
          if (zoomed) e.preventDefault();
          return;
      }

      // undo/redo of a geometry edit
      if ((e.metaKey || e.ctrlKey) && e.code === "KeyZ") {
        e.preventDefault();
        void (e.shiftKey ? redo() : undo());
        return;
      }

      // digits/letters by e.code: they work with CapsLock/Shift and on any layout
      if (e.metaKey || e.ctrlKey || e.altKey) return;

      // keyboard zoom: 0 — bring back the whole frame
      if (e.code === "Digit0" || e.code === "Numpad0") {
        resetView();
        return;
      }

      // 1-9 — class of the selected annotation (and class for the next boxes);
      // code so it works on any layout, key as the fallback
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
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.key === " ") spaceHeld.current = false;
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [
    filtered,
    selectedId,
    navigateSibling,
    undo,
    redo,
    resetView,
    zoomed,
    setStatusAndAdvance,
    remove,
    classes,
    isOcr,
    pickClass,
    canDraw,
  ]);

  // --- coordinates, zoom, drawing and editing ---
  /** Client coordinates -> image coordinates, taking the zoom into account. */
  const toImageCoords = (e: { clientX: number; clientY: number }) => {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect || !image || !view) return { x: 0, y: 0 };
    return {
      x: view.x + ((e.clientX - rect.left) / rect.width) * view.w,
      y: view.y + ((e.clientY - rect.top) / rect.height) * view.h,
    };
  };

  /** Wheel zoom towards the point under the cursor, so that it stays put. */
  const onWheel = (e: React.WheelEvent) => {
    if (!image || !view) return;
    const factor = e.deltaY < 0 ? 1 / 1.15 : 1.15;
    const w = Math.max(image.width / ZOOM_MAX, Math.min(view.w * factor, image.width));
    const scale = w / view.w;
    if (scale === 1) return;
    const p = toImageCoords(e);
    const h = view.h * scale;
    setView({
      w,
      h,
      x: Math.max(0, Math.min(p.x - (p.x - view.x) * scale, image.width - w)),
      y: Math.max(0, Math.min(p.y - (p.y - view.y) * scale, image.height - h)),
    });
  };

  /** Save the geometry and push the edit onto the undo stack. */
  const commitGeometry = useCallback(
    async (id: string, before: Geometry, after: Geometry) => {
      try {
        await patch(id, { geometry: after });
      } catch (err) {
        setError(errorText(err));
        // the server refused it — put the box back so the screen does not lie
        setAnnotations((as) =>
          as.map((a) => (a.id === id ? { ...a, geometry: before } : a))
        );
        return;
      }
      setError(null);
      undoStack.current.push({ id, before, after });
      redoStack.current = [];
      setUndoDepth(undoStack.current.length);
    },
    [patch]
  );

  const startResize = (e: React.PointerEvent, a: Annotation, handle: Handle) => {
    const box = asBox(a.geometry);
    if (!box) return;
    e.stopPropagation();
    (e.target as Element).setPointerCapture?.(e.pointerId);
    setSelectedId(a.id);
    setDrag({ mode: "resize", id: a.id, handle, before: box, box });
  };

  const startMove = (e: React.PointerEvent, a: Annotation) => {
    const box = asBox(a.geometry);
    if (!box || drawMode) return;
    e.stopPropagation();
    (e.target as Element).setPointerCapture?.(e.pointerId);
    setSelectedId(a.id);
    setDrag({ mode: "move", id: a.id, before: box, box, grab: toImageCoords(e) });
  };

  const onPointerDown = (e: React.PointerEvent) => {
    if (drawMode) {
      drawStart.current = toImageCoords(e);
      setDraftRect({ ...drawStart.current, w: 0, h: 0 });
      return;
    }
    // pan: middle button, or space held down while zoomed in
    if (view && (e.button === 1 || spaceHeld.current) && zoomed) {
      e.preventDefault();
      setDrag({
        mode: "pan",
        startPointer: { x: e.clientX, y: e.clientY },
        startView: view,
      });
    }
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (drawMode && drawStart.current) {
      const p = toImageCoords(e);
      setDraftRect({
        x: Math.min(drawStart.current.x, p.x),
        y: Math.min(drawStart.current.y, p.y),
        w: Math.abs(p.x - drawStart.current.x),
        h: Math.abs(p.y - drawStart.current.y),
      });
      return;
    }
    if (!drag || !image) return;

    if (drag.mode === "pan") {
      const rect = svgRef.current?.getBoundingClientRect();
      if (!rect) return;
      const dx = ((e.clientX - drag.startPointer.x) / rect.width) * drag.startView.w;
      const dy = ((e.clientY - drag.startPointer.y) / rect.height) * drag.startView.h;
      setView({
        ...drag.startView,
        x: Math.max(
          0,
          Math.min(drag.startView.x - dx, image.width - drag.startView.w)
        ),
        y: Math.max(
          0,
          Math.min(drag.startView.y - dy, image.height - drag.startView.h)
        ),
      });
      return;
    }

    const p = toImageCoords(e);
    const box =
      drag.mode === "resize"
        ? resizeBox(drag.before, drag.handle, p, image.width, image.height)
        : moveBox(
            drag.before,
            p.x - drag.grab.x,
            p.y - drag.grab.y,
            image.width,
            image.height
          );
    setDrag({ ...drag, box });
    // draw optimistically: the box must follow the cursor with no network lag
    setAnnotations((as) =>
      as.map((a) => (a.id === drag.id ? { ...a, geometry: boxGeometry(box) } : a))
    );
  };

  const onPointerUp = async () => {
    if (drag) {
      const finished = drag;
      setDrag(null);
      if (finished.mode !== "pan") {
        const after = boxGeometry(finished.box);
        const before = boxGeometry(finished.before);
        // a click with no movement is not an edit — it would clutter the undo stack
        if (JSON.stringify(after) !== JSON.stringify(before)) {
          await commitGeometry(finished.id, before, after);
        }
      }
      return;
    }
    if (!drawMode || !draftRect) return;
    drawStart.current = null;
    const rect = draftRect;
    setDraftRect(null);
    setDrawMode(false);
    if (rect.w < 8 || rect.h < 8) return; // stray click
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

  if (!image) return <p className="muted">{error ?? "Loading..."}</p>;

  const imageIndex = images.findIndex((i) => i.id === imageId);
  const done = annotations.filter(
    (a) => a.status === "accepted" || a.status === "edited"
  ).length;
  // SVG labels live in image coordinates — so their size is derived from it
  const labelSize = Math.max(10, Math.round(Math.max(image.width, image.height) / 55));
  // handles are sized in image coordinates but scaled back by the zoom, so that
  // at any zoom level they stay the same size on screen
  const handleSize = ((view?.w ?? image.width) / image.width) * (labelSize * 0.8);
  const activeColor = activeClass
    ? classByName.get(activeClass)?.color ?? UNKNOWN_CLASS_COLOR
    : "#2f7df6";

  return (
    <div>
      <div className="toolbar">
        <Link to={`/projects/${projectId}`}>← back to project</Link>
        <span className="muted">
          image {imageIndex + 1} / {images.length} · reviewed {done}/
          {annotations.length}
        </span>
        <button
          className={drawMode ? "active" : ""}
          onClick={() => setDrawMode((v) => !v)}
          disabled={!canDraw}
          title={
            canDraw
              ? "D"
              : "Add project classes first — a new box needs a class"
          }
        >
          + box
          {!isOcr && activeClass ? `: ${activeClass}` : ""}
        </button>
        <button onClick={() => void undo()} disabled={undoDepth === 0} title="Cmd+Z">
          ↶ undo
        </button>
        {/* An image with no objects would otherwise never reach the dataset:
            there is nothing to label, yet the detector needs background examples */}
        {annotations.length === 0 && (
          <button
            className={image?.reviewed ? "active" : ""}
            onClick={() => void markEmpty()}
            title="Mark this image as checked and empty — it becomes a background example in the dataset"
          >
            {image?.reviewed ? "✓ marked empty" : "nothing here"}
          </button>
        )}
        {zoomed && (
          <button onClick={resetView} title="0">
            ⤢ fit image
          </button>
        )}
        <span className="zoom-hint muted">
          {zoomed
            ? "wheel — zoom, space+drag — pan"
            : "wheel — zoom"}
        </span>
        <span className="toolbar-group">
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value as StatusFilter)}
          >
            <option value="all">all statuses</option>
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
            viewBox={
              view
                ? `${view.x} ${view.y} ${view.w} ${view.h}`
                : `0 0 ${image.width} ${image.height}`
            }
            className={drawMode ? "drawing" : drag?.mode === "pan" ? "panning" : ""}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={() => void onPointerUp()}
            onPointerCancel={() => setDrag(null)}
            onWheel={onWheel}
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
                      onPointerDown={(e) => startMove(e, a)}
                      style={{
                        cursor: drawMode
                          ? "crosshair"
                          : isSelected
                            ? "move"
                            : "pointer",
                      }}
                    />
                  )}
                  {/* handles only on the selected box — otherwise the screen is noisy */}
                  {isSelected &&
                    !drawMode &&
                    a.geometry.type === "bbox" &&
                    HANDLES.map((h) => {
                      const g = a.geometry as Extract<Geometry, { type: "bbox" }>;
                      const hx =
                        h.includes("w")
                          ? g.x
                          : h.includes("e")
                            ? g.x + g.width
                            : g.x + g.width / 2;
                      const hy =
                        h.includes("n")
                          ? g.y
                          : h.includes("s")
                            ? g.y + g.height
                            : g.y + g.height / 2;
                      return (
                        <rect
                          key={h}
                          className="box-handle"
                          x={hx - handleSize / 2}
                          y={hy - handleSize / 2}
                          width={handleSize}
                          height={handleSize}
                          fill={color}
                          onPointerDown={(e) => startResize(e, a, h)}
                          style={{ cursor: HANDLE_CURSOR[h] }}
                        />
                      );
                    })}
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
                Classes <span className="muted">(1-9 — assign)</span>
              </div>
              {classes.length === 0 ? (
                <p className="muted">
                  No classes yet —{" "}
                  <Link to={`/projects/${projectId}`}>add them in the project</Link>.
                </p>
              ) : (
                <ul>
                  {classes.map((c, i) => (
                    <li
                      key={c.id}
                      className={c.name === activeClass ? "active" : ""}
                      onClick={() => pickClass(c.name)}
                      title="assign to the selected annotation"
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
                  class
                  <select
                    value={selected.label}
                    disabled={classes.length === 0}
                    onChange={(e) => void assignClass(selected.id, e.target.value)}
                  >
                    {!classByName.has(selected.label) && (
                      <option value={selected.label}>
                        {selected.label || "—"} — not in the project
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
                {isOcr && <button onClick={() => void saveDraft()}>Save</button>}
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
            <p className="hint">
              Click an annotation. Keys: ↑↓ — select, ←→ — prev/next image, A —
              accept, R — reject, {isOcr ? "E — text, " : "1-9 — class, "}D —
              new box, Del — delete.
            </p>
          )}

          <ul className="ann-list ann-list-card">
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
