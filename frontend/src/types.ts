export type ImageRecord = {
  id: number;
  file_name: string;
  width: number;
  height: number;
};

export type ExportWorkspaceImageRecord = ImageRecord & {
  annotation_count: number;
};

export type ExportWorkspaceFolderRecord = {
  folder: string;
  image_count: number;
  annotated_image_count: number;
  annotation_count: number;
  images: ExportWorkspaceImageRecord[];
};

export type ExportWorkspaceResponse = {
  root: string;
  folders: ExportWorkspaceFolderRecord[];
};

export type ExportSplit = "train" | "val" | "test";

export type ExportCocoRequest = {
  root?: string;
  splits?: Record<ExportSplit, number[]>;
  folder_splits?: Array<{
    folder: string;
    splits: Record<ExportSplit, number[]>;
  }>;
};

export type ExportCocoResponse = {
  export_dir: string;
  coco_json: string;
  mask_count: number;
  issues: unknown[];
  split_coco_jsons: Record<string, string>;
  folder_exports: Array<{
    folder: string;
    export_dir: string;
    coco_json: string;
    mask_count: number;
    split_coco_jsons: Record<string, string>;
  }>;
};

export type ImagePage = {
  items: ImageRecord[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
};

export type ProjectIndexStatus = {
  status: "idle" | "indexing" | "completed" | "failed";
  indexed_count: number;
  total_seen: number | null;
  current_file: string | null;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
};

export type AnnotationRecord = {
  id: number;
  image_id: number;
  category_id: number | null;
  category_name: string | null;
  bbox: number[];
  area: number;
  iscrowd: number;
  mask_path: string;
  version: number;
  status: string;
  score: number | null;
  visible: boolean;
};

export type MaskCandidate = {
  mask_png: string;
  bbox: number[];
  area: number;
  score: number | null;
  prompt_type: string;
  annotation: AnnotationRecord | null;
};

export type CandidateFilterMode = "top_k" | "threshold_top_k";

export type ToolMode =
  | "view"
  | "point_pos"
  | "point_neg"
  | "box"
  | "text"
  | "brush"
  | "erase"
  | "polygon";

export type SearchResult = {
  image: ImageRecord;
  score: number;
};

export type ClipStatus = {
  available: boolean;
  indexed: number;
  total: number;
  model: string;
  device: string;
  error: string | null;
};

export type BulkMode = "all" | "clip_filtered";

export type BulkJobRecord = {
  id: number;
  kind: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  payload: Record<string, unknown>;
  result: {
    processed?: number;
    total?: number;
    created?: number;
    cancel_requested?: boolean;
    error?: string | null;
  };
  created_at: string;
  updated_at: string;
};

export type ReviewCandidateRecord = {
  id: number;
  job_id: number;
  image: ImageRecord;
  category_name: string;
  prompt_text: string;
  mask_path: string;
  bbox: number[];
  area: number;
  score: number | null;
  rank: number;
  status: "pending" | "accepted" | "rejected";
  annotation_id: number | null;
  created_at: string;
};
