/** Project task type. Object detection is the main case, OCR the secondary one. */
export type TaskType = "detection" | "ocr";

export interface Project {
  id: string;
  name: string;
  description: string;
  task_type: TaskType;
  created_at: string;
}

/** Project class: the name is always English (the engine gets it as a text query). */
export interface ProjectClass {
  id: string;
  name: string;
  color: string;
  sort_order: number;
}

export interface ImageListItem {
  id: string;
  page_index: number;
  width: number;
  height: number;
  total_annotations: number;
  pending_annotations: number;
  created_at: string;
}

export interface ImageMeta {
  id: string;
  document_id: string;
  page_index: number;
  width: number;
  height: number;
  blur_score: number | null;
  /** A human confirmed they looked at the image (needed for images with no objects). */
  reviewed: boolean;
}

export type Geometry =
  | { type: "bbox"; x: number; y: number; width: number; height: number }
  | { type: "polygon"; points: [number, number][] };

export type AnnotationStatus = "pending" | "accepted" | "edited" | "rejected";

export interface Annotation {
  id: string;
  image_id: string;
  geometry: Geometry;
  /** For detection — a project class name; for ocr — the line type (text_line). */
  label: string;
  text: string | null;
  confidence: number;
  source: { type: string; name: string; version: string | null };
  status: AnnotationStatus;
}

export interface Job {
  id: string;
  status: "queued" | "running" | "done" | "failed";
  result: Record<string, unknown>;
}

export type GpuStatus = "not_configured" | "deploying" | "ready" | "failed";

/** One GPU recipe deployed (or not) into the user's Modal account. */
export interface GpuDeployment {
  /** Engine name it is served under: modal_gpu (OCR), modal_gpu_detect (boxes). */
  engine: string;
  title: string;
  task: TaskType;
  status: GpuStatus;
  endpoint_url: string | null;
  error: string | null;
}

export interface Settings {
  access_protected: boolean;
  modal_configured: boolean;
  modal_token_id_masked: string | null;
  /** Legacy mirror of the OCR GPU (engine modal_gpu). Prefer `gpus`. */
  gpu_status: GpuStatus;
  gpu_endpoint_url: string | null;
  gpu_error: string | null;
  /** One entry per GPU recipe, in registry order. */
  gpus: GpuDeployment[];
}

export type LabelerRequires = "cpu" | "modal" | "config";

export interface Labeler {
  name: string;
  title: string;
  requires: LabelerRequires;
  available: boolean;
  reason: string | null;
  /** Task types the engine handles. If the server omits the field, it is universal. */
  tasks?: TaskType[];
}

/** An engine fits a project if it explicitly lists that task type (or lists none). */
export const labelerSupportsTask = (labeler: Labeler, task: TaskType): boolean =>
  !labeler.tasks || labeler.tasks.includes(task);

export const TASK_TITLES: Record<TaskType, string> = {
  detection: "Object detection",
  ocr: "OCR (text)",
};

/** Default engine per task: owlv2 was chosen by measurement (F1 0.823 on 79 photos). */
export const DEFAULT_LABELER: Record<TaskType, string> = {
  detection: "owlv2",
  ocr: "rapidocr",
};

/** Engine threshold: by default boxes below 0.25 are not shown to the human. */
export const DEFAULT_SCORE_THRESHOLD = 0.25;

/** Bulk-accept threshold: at 0.40 the precision of the predictions is 0.94. */
export const DEFAULT_ACCEPT_THRESHOLD: Record<TaskType, number> = {
  detection: 0.4,
  ocr: 0.9,
};

export const EXPORT_FORMATS: Record<TaskType, { value: string; title: string }[]> = {
  detection: [
    { value: "yolo_detect", title: "YOLO detect" },
    { value: "coco", title: "COCO" },
  ],
  ocr: [
    { value: "paddleocr_det", title: "PaddleOCR det" },
    { value: "paddleocr_rec", title: "PaddleOCR rec" },
    { value: "coco", title: "COCO" },
  ],
};

const BASE = "/api/v1";

export class ApiError extends Error {
  readonly status: number;
  readonly body: string;
  readonly payload: unknown;

  constructor(status: number, body: string) {
    let payload: unknown = null;
    try {
      payload = JSON.parse(body);
    } catch {
      payload = null;
    }
    super(ApiError.describe(status, body, payload));
    this.name = "ApiError";
    this.status = status;
    this.body = body;
    this.payload = payload;
  }

  /** FastAPI puts the human text in detail (a string, or an object with message). */
  private static describe(status: number, body: string, payload: unknown): string {
    const detail =
      payload && typeof payload === "object"
        ? (payload as Record<string, unknown>).detail
        : undefined;
    if (typeof detail === "string" && detail.trim()) return detail;
    if (detail && typeof detail === "object") {
      const message = (detail as Record<string, unknown>).message;
      if (typeof message === "string" && message.trim()) return message;
    }
    return `API ${status}: ${body}`;
  }
}

export const errorText = (err: unknown): string =>
  err instanceof Error ? err.message : String(err);

const firstNumber = (value: unknown, depth = 0): number | null => {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const m = value.match(/\d+/);
    return m ? Number(m[0]) : null;
  }
  if (value && typeof value === "object" && depth < 3) {
    for (const nested of Object.values(value as Record<string, unknown>)) {
      const n = firstNumber(nested, depth + 1);
      if (n !== null) return n;
    }
  }
  return null;
};

/**
 * DELETE /classes/{id} on a class still in use answers 409 with the annotation
 * count. We do not pin the exact body shape — we just pull out the first number.
 */
export const annotationsBlockingDelete = (err: unknown): number | null => {
  if (!(err instanceof ApiError) || err.status !== 409) return null;
  return firstNumber(err.payload) ?? firstNumber(err.body) ?? 0;
};

async function requestVoid(path: string, init?: RequestInit): Promise<void> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    throw new ApiError(res.status, await res.text());
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    throw new ApiError(res.status, await res.text());
  }
  return res.json();
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  // projects
  listProjects: () => request<Project[]>("/projects"),
  createProject: (name: string, taskType: TaskType = "detection") =>
    request<Project>("/projects", json("POST", { name, task_type: taskType })),
  getProject: (id: string) => request<Project>(`/projects/${id}`),

  // project classes
  listClasses: (projectId: string) =>
    request<ProjectClass[]>(`/projects/${projectId}/classes`),
  createClass: (projectId: string, name: string, color: string | null = null) =>
    request<ProjectClass>(
      `/projects/${projectId}/classes`,
      json("POST", { name, color })
    ),
  /** Replace the whole list: the server assigns the colors. */
  replaceClasses: (projectId: string, names: string[]) =>
    request<ProjectClass[]>(`/projects/${projectId}/classes`, json("PUT", { names })),
  updateClass: (
    classId: string,
    patch: Partial<Pick<ProjectClass, "name" | "color" | "sort_order">>
  ) => request<ProjectClass>(`/classes/${classId}`, json("PATCH", patch)),
  deleteClass: (classId: string, force = false) =>
    requestVoid(`/classes/${classId}${force ? "?force=true" : ""}`, {
      method: "DELETE",
    }),

  // documents / images
  uploadDocument: (projectId: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<{ id: string }>(`/projects/${projectId}/documents`, {
      method: "POST",
      body: form,
    });
  },
  listProjectImages: (projectId: string) =>
    request<ImageListItem[]>(`/projects/${projectId}/images`),
  getImage: (id: string) => request<ImageMeta>(`/images/${id}`),
  getImageUrl: (id: string) =>
    request<{ url: string }>(`/images/${id}/url`).then((r) => r.url),

  // annotations
  listAnnotations: (imageId: string) =>
    request<Annotation[]>(`/images/${imageId}/annotations`),
  createAnnotation: (
    imageId: string,
    geometry: Geometry,
    init: { label?: string; text?: string } = {}
  ) =>
    request<Annotation>(
      `/images/${imageId}/annotations`,
      json("POST", { geometry, ...init })
    ),
  updateAnnotation: (
    id: string,
    patch: Partial<Pick<Annotation, "text" | "status" | "label" | "geometry">>
  ) => request<Annotation>(`/annotations/${id}`, json("PATCH", patch)),
  deleteAnnotation: (id: string) =>
    requestVoid(`/annotations/${id}`, { method: "DELETE" }),
  bulkAccept: (projectId: string, minConfidence: number) =>
    request<{ accepted: number }>(
      `/projects/${projectId}/annotations/bulk-accept`,
      json("POST", { min_confidence: minConfidence })
    ),

  // settings / labelers
  getSettings: () => request<Settings>("/settings"),
  saveModalToken: (tokenId: string, tokenSecret: string) =>
    request<Settings>(
      "/settings",
      json("PUT", { modal_token_id: tokenId, modal_token_secret: tokenSecret })
    ),
  deleteModalToken: () =>
    request<Settings>("/settings/modal", { method: "DELETE" }),
  /** Deploy one GPU recipe (engine name from settings.gpus). */
  deployGpu: (engine: string) =>
    request<Job>("/settings/gpu/deploy", json("POST", { engine })),
  listLabelers: () => request<Labeler[]>("/labelers"),

  // jobs
  startAutolabel: (
    projectId: string,
    labeler: string | null,
    config: Record<string, unknown>,
    // rerun over already-labeled images: needed when the class list changed
    rerun = false
  ) =>
    request<Job>(
      `/projects/${projectId}/autolabel`,
      json("POST", { labeler, config, rerun })
    ),
  /** Mark an image reviewed: an empty image then enters the dataset as background. */
  markImageReviewed: (imageId: string, reviewed: boolean) =>
    request<ImageMeta>(`/images/${imageId}`, json("PATCH", { reviewed })),

  getJob: (id: string) => request<Job>(`/jobs/${id}`),
  // the project's unfinished job: the page picks it up again after a reload
  getActiveJob: (projectId: string) =>
    request<Job | null>(`/projects/${projectId}/jobs/active`),
};
