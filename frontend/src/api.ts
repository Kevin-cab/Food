import type { AnnotationRecord, BulkJobRecord, BulkMode, ClipStatus, ExportCocoRequest, ExportCocoResponse, ExportWorkspaceResponse, ImageCountOperator, ImageMaskFilter, ImagePage, MaskCandidate, ProjectIndexStatus, ReviewCandidateRecord, SearchResult } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    }
  });
  if (!response.ok) {
    const message = await response.text();
    let detail = "";
    try {
      const parsed = JSON.parse(message);
      if (typeof parsed.detail === "string") {
        detail = parsed.detail;
      } else if (Array.isArray(parsed.detail)) {
        detail = parsed.detail
          .map((item: { loc?: unknown[]; msg?: string }) => {
            const location = Array.isArray(item.loc) ? item.loc.join(".") : "request";
            return `${location}: ${item.msg ?? JSON.stringify(item)}`;
          })
          .join("; ");
      } else if (parsed.detail) {
        detail = JSON.stringify(parsed.detail);
      }
    } catch {
      detail = "";
    }
    throw new Error(detail || message || response.statusText);
  }
  return response.json() as Promise<T>;
}

const cacheKey = (projectKey?: string) => projectKey ? `?project_key=${encodeURIComponent(projectKey)}` : "";

export const imageUrl = (id: number, projectKey?: string) => `${API_BASE}/api/images/${id}/file${cacheKey(projectKey)}`;
export const annotationMaskUrl = (id: number, projectKey?: string) => `${API_BASE}/api/annotations/${id}/mask${cacheKey(projectKey)}`;
export const reviewCandidateMaskUrl = (id: number, projectKey?: string) => `${API_BASE}/api/bulk/candidates/${id}/mask${cacheKey(projectKey)}`;

export const api = {
  openProject: (path: string) =>
    request<{ root: string; db_path: string; image_count: number }>("/api/projects/open", {
      method: "POST",
      body: JSON.stringify({ path })
    }),
  exportWorkspaceScan: (path: string) =>
    request<ExportWorkspaceResponse>("/api/export/workspace/scan", {
      method: "POST",
      body: JSON.stringify({ path })
    }),
  projectIndexStatus: () => request<ProjectIndexStatus>("/api/projects/index-status"),
  listImages: (params: { limit?: number; offset?: number; q?: string; mask_filter?: ImageMaskFilter; class_name?: string; count_op?: ImageCountOperator; count_value?: number | null } = {}) => {
    const search = new URLSearchParams();
    search.set("limit", String(params.limit ?? 200));
    search.set("offset", String(params.offset ?? 0));
    if (params.q) search.set("q", params.q);
    if (params.mask_filter && params.mask_filter !== "all") search.set("mask_filter", params.mask_filter);
    if (params.class_name) search.set("class_name", params.class_name);
    if (params.count_op) search.set("count_op", params.count_op);
    if (params.count_value != null) search.set("count_value", String(params.count_value));
    return request<ImagePage>(`/api/images?${search.toString()}`);
  },
  listAnnotations: (imageId: number) =>
    request<AnnotationRecord[]>(`/api/images/${imageId}/annotations`),
  createAnnotation: (imageId: number, categoryName: string, maskPng: string, status = "accepted", score?: number | null) =>
    request<AnnotationRecord>("/api/annotations", {
      method: "POST",
      body: JSON.stringify({ image_id: imageId, category_name: categoryName, mask_png: maskPng, status, score })
    }),
  updateAnnotation: (id: number, changes: Partial<Pick<AnnotationRecord, "category_name" | "visible" | "status">>) =>
    request<AnnotationRecord>(`/api/annotations/${id}`, {
      method: "PATCH",
      body: JSON.stringify(changes)
    }),
  deleteAnnotation: (id: number) =>
    request<{ deleted: number }>(`/api/annotations/${id}`, { method: "DELETE" }),
  replaceMask: (id: number, maskPng: string) =>
    request<AnnotationRecord>(`/api/annotations/${id}/mask`, {
      method: "POST",
      body: JSON.stringify({ mask_png: maskPng })
    }),
  undoAnnotation: (id: number) =>
    request<AnnotationRecord>(`/api/annotations/${id}/undo`, { method: "POST" }),
  redoAnnotation: (id: number) =>
    request<AnnotationRecord>(`/api/annotations/${id}/redo`, { method: "POST" }),
  samPrompt: (payload: unknown) =>
    request<{ candidates: MaskCandidate[]; sam_error: string | null }>("/api/sam/prompt", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  clipStatus: () => request<ClipStatus>("/api/clip/status"),
  indexClip: () =>
    request<ClipStatus & { indexed: number }>("/api/clip/index", {
      method: "POST",
      body: JSON.stringify({ force: false })
    }),
  rebuildClip: () =>
    request<ClipStatus & { indexed: number }>("/api/clip/index", {
      method: "POST",
      body: JSON.stringify({ force: true })
    }),
  searchClip: (payload: { text?: string; image_id?: number; limit?: number }) =>
    request<SearchResult[]>("/api/clip/search", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  createBulkJob: (payload: {
    mode: BulkMode;
    text: string;
    category_name?: string;
    confidence_threshold: number;
    top_k: number;
    max_images?: number;
    clip_image_ids?: number[];
  }) =>
    request<BulkJobRecord>("/api/bulk/jobs", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  listBulkJobs: () => request<BulkJobRecord[]>("/api/bulk/jobs?limit=20"),
  getBulkJob: (id: number) => request<BulkJobRecord>(`/api/bulk/jobs/${id}`),
  cancelBulkJob: (id: number) =>
    request<BulkJobRecord>(`/api/bulk/jobs/${id}/cancel`, { method: "POST" }),
  deleteBulkJob: (id: number) =>
    request<{ deleted: number }>(`/api/bulk/jobs/${id}`, { method: "DELETE" }),
  deleteFinishedBulkJobs: () =>
    request<{ deleted: number[] }>("/api/bulk/jobs/finished", { method: "DELETE" }),
  listReviewCandidates: (jobId: number, status = "pending") =>
    request<ReviewCandidateRecord[]>(`/api/bulk/jobs/${jobId}/candidates?status=${encodeURIComponent(status)}`),
  openReviewCandidate: (id: number) =>
    request<{ candidate: ReviewCandidateRecord; mask_png: string }>(`/api/bulk/candidates/${id}/open`),
  updateReviewCandidate: (id: number, categoryName: string) =>
    request<ReviewCandidateRecord>(`/api/bulk/candidates/${id}`, {
      method: "PATCH",
      body: JSON.stringify({ category_name: categoryName })
    }),
  renamePendingReviewCandidates: (jobId: number, categoryName: string) =>
    request<{ updated: number; ids: number[]; category_name: string }>(`/api/bulk/jobs/${jobId}/candidates`, {
      method: "PATCH",
      body: JSON.stringify({ category_name: categoryName })
    }),
  acceptPendingReviewCandidates: (jobId: number) =>
    request<{ accepted: number; ids: number[]; annotation_ids: number[] }>(`/api/bulk/jobs/${jobId}/candidates/accept`, {
      method: "POST"
    }),
  acceptReviewCandidate: (id: number) =>
    request<ReviewCandidateRecord>(`/api/bulk/candidates/${id}/accept`, { method: "POST" }),
  linkAcceptedReviewCandidate: (id: number, annotationId: number) =>
    request<ReviewCandidateRecord>(`/api/bulk/candidates/${id}/link-accepted`, {
      method: "POST",
      body: JSON.stringify({ annotation_id: annotationId })
    }),
  rejectReviewCandidate: (id: number) =>
    request<ReviewCandidateRecord>(`/api/bulk/candidates/${id}/reject`, { method: "POST" }),
  reopenReviewCandidate: (id: number) =>
    request<ReviewCandidateRecord>(`/api/bulk/candidates/${id}/reopen`, { method: "POST" }),
  bulkConcept: (text: string, limit: number) =>
    request<{ annotations: AnnotationRecord[]; matches: SearchResult[] }>("/api/bulk/concept", {
      method: "POST",
      body: JSON.stringify({ text, category_name: text, limit, accept: false })
    }),
  propagate: (annotationId: number) =>
    request<{ annotations: AnnotationRecord[]; matches: SearchResult[] }>("/api/propagate", {
      method: "POST",
      body: JSON.stringify({ annotation_id: annotationId, limit: 24, run_sam: true })
    }),
  validate: () => request<Array<{ severity: string; code: string; message: string }>>("/api/qa/validate"),
  exportCoco: (payload?: ExportCocoRequest) =>
    request<ExportCocoResponse>("/api/export/coco", {
      method: "POST",
      body: JSON.stringify(payload ?? {})
    })
};
