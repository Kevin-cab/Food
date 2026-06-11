import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Box,
  Brush,
  Check,
  ChevronLeft,
  ChevronRight,
  Download,
  Eye,
  EyeOff,
  Hand,
  Maximize2,
  MousePointer2,
  Plus,
  Redo2,
  Search,
  Sparkles,
  Trash2,
  Type,
  Undo2,
  Wand2,
  ZoomIn,
  ZoomOut,
  X
} from "lucide-react";
import { annotationMaskUrl, api, imageUrl } from "./api";
import type { AnnotationRecord, BulkJobRecord, BulkMode, CandidateFilterMode, ClipStatus, ExportCocoRequest, ExportSplit, ImageCountOperator, ImageMaskFilter, ImageRecord, MaskCandidate, ProjectIndexStatus, ReviewCandidateRecord, SearchResult, ToolMode } from "./types";
import "./styles.css";

type Point = { x: number; y: number };
type PromptPoint = Point & { label: 0 | 1 };
type Candidate = MaskCandidate & { localId: number; imageId: number; name: string; visible: boolean; reviewCandidateId?: number };
type ExportSplitPlan = {
  train: ImageRecord[];
  val: ImageRecord[];
  test: ImageRecord[];
};
type ExportFolderRecord = {
  folder: string;
  images: ImageRecord[];
  annotatedImages: ImageRecord[];
  annotationCount: number;
};
type ExportBucketEntry = {
  folder: string;
  split: ExportSplit;
  images: ImageRecord[];
};
type ExportGroupRecord = {
  id: number;
  name: string;
  folders: string[];
};
type ResizeHandle = "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w";
type ResizeDrag = {
  target: "candidate" | "annotation";
  id: number;
  handle: ResizeHandle;
  originalBbox: number[];
  originalMask: string;
};
type WorkState = {
  candidates: Candidate[];
  selectedCandidate: number | null;
  promptPoints: PromptPoint[];
  polygonPoints: Point[];
};
type EditorAction = {
  label: string;
  undo: () => Promise<void>;
  redo?: () => Promise<void>;
};
type HistoryKind = "candidate" | "editor";

const COLORS = [
  "#e6194b",
  "#3cb44b",
  "#ffe119",
  "#0082c8",
  "#f58231",
  "#911eb4",
  "#46f0f0",
  "#f032e6",
  "#d2f53c",
  "#fabebe"
];

const IMAGE_PAGE_SIZE = 200;
const DEFAULT_PROJECT_PATH = "sam3/assets/images";
const DEFAULT_EXPORT_ROOT = "E:\\nutrition5k_dataset\\tw_food_101\\tw_food_101\\train";

const TOOL_ITEMS: Array<{ id: ToolMode; label: string; icon: React.ReactNode }> = [
  { id: "view", label: "View / pan", icon: <Hand size={17} /> },
  { id: "point_pos", label: "Positive point", icon: <Plus size={17} /> },
  { id: "point_neg", label: "Negative point", icon: <X size={17} /> },
  { id: "box", label: "Box prompt", icon: <Box size={17} /> },
  { id: "text", label: "Text prompt", icon: <Type size={17} /> },
  { id: "brush", label: "Brush add mask pixels", icon: <Brush size={17} /> },
  { id: "erase", label: "Erase mask pixels", icon: <X size={17} /> },
  { id: "polygon", label: "Manual polygon", icon: <MousePointer2 size={17} /> }
];

function App() {
  const [projectPath, setProjectPath] = useState(DEFAULT_PROJECT_PATH);
  const [exportRoot, setExportRoot] = useState(DEFAULT_EXPORT_ROOT);
  const [project, setProject] = useState<string>("");
  const [projectKey, setProjectKey] = useState<string>("");
  const [images, setImages] = useState<ImageRecord[]>([]);
  const [imageTotal, setImageTotal] = useState(0);
  const [annotationClassName, setAnnotationClassName] = useState("food");
  const [imageFiltersOpen, setImageFiltersOpen] = useState(false);
  const [imageFilenameFilter, setImageFilenameFilter] = useState("");
  const [imageMaskFilter, setImageMaskFilter] = useState<ImageMaskFilter>("all");
  const [imageClassFilter, setImageClassFilter] = useState("");
  const [imageCountOp, setImageCountOp] = useState<ImageCountOperator>("gte");
  const [imageCountValue, setImageCountValue] = useState("");
  const [imageLoading, setImageLoading] = useState(false);
  const [imageHasMore, setImageHasMore] = useState(false);
  const [indexStatus, setIndexStatus] = useState<ProjectIndexStatus | null>(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [annotations, setAnnotations] = useState<AnnotationRecord[]>([]);
  const [selectedAnnotation, setSelectedAnnotation] = useState<number | null>(null);
  const [tool, setTool] = useState<ToolMode>("view");
  const [promptText, setPromptText] = useState("food");
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [selectedCandidate, setSelectedCandidate] = useState<number | null>(null);
  const [promptPoints, setPromptPoints] = useState<PromptPoint[]>([]);
  const [polygonPoints, setPolygonPoints] = useState<Point[]>([]);
  const [candidateHistory, setCandidateHistory] = useState<WorkState[]>([]);
  const [candidateRedo, setCandidateRedo] = useState<WorkState[]>([]);
  const [editorHistory, setEditorHistory] = useState<EditorAction[]>([]);
  const [editorRedo, setEditorRedo] = useState<EditorAction[]>([]);
  const [undoKinds, setUndoKinds] = useState<HistoryKind[]>([]);
  const [redoKinds, setRedoKinds] = useState<HistoryKind[]>([]);
  const [annotationMasks, setAnnotationMasks] = useState<Record<number, string>>({});
  const [status, setStatus] = useState("Open a folder to begin.");
  const [rightTab, setRightTab] = useState<"editor" | "bulk" | "export">("editor");
  const [searchText, setSearchText] = useState("noodles");
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [bulkPrompt, setBulkPrompt] = useState("bowl");
  const [bulkRenameText, setBulkRenameText] = useState("");
  const [bulkMode, setBulkMode] = useState<BulkMode>("all");
  const [bulkThreshold, setBulkThreshold] = useState(0.3);
  const [bulkTopK, setBulkTopK] = useState(3);
  const [bulkMaxImages, setBulkMaxImages] = useState(50);
  const [promptFilterMode, setPromptFilterMode] = useState<CandidateFilterMode>("top_k");
  const [promptThreshold, setPromptThreshold] = useState(0.3);
  const [promptTopK, setPromptTopK] = useState(5);
  const [clipStatus, setClipStatus] = useState<ClipStatus | null>(null);
  const [bulkJobs, setBulkJobs] = useState<BulkJobRecord[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<number | null>(null);
  const [reviewCandidates, setReviewCandidates] = useState<ReviewCandidateRecord[]>([]);
  const [pendingReviewOpen, setPendingReviewOpen] = useState<{ candidate: ReviewCandidateRecord; mask_png: string } | null>(null);
  const [boxStart, setBoxStart] = useState<Point | null>(null);
  const [draftBox, setDraftBox] = useState<{ x: number; y: number; width: number; height: number } | null>(null);
  const [editMaskData, setEditMaskData] = useState<string | null>(null);
  const [showCanvasLabels, setShowCanvasLabels] = useState(true);
  const [painting, setPainting] = useState(false);
  const [brushSize, setBrushSize] = useState(18);
  const [brushCursor, setBrushCursor] = useState<Point | null>(null);
  const [editingTarget, setEditingTarget] = useState<"candidate" | "annotation" | null>(null);
  const [dragPromptIndex, setDragPromptIndex] = useState<number | null>(null);
  const [dragPolygonIndex, setDragPolygonIndex] = useState<number | null>(null);
  const [resizeDrag, setResizeDrag] = useState<ResizeDrag | null>(null);
  const [fitScale, setFitScale] = useState(1);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState<Point>({ x: 0, y: 0 });
  const [panning, setPanning] = useState(false);
  const [exportFolders, setExportFolders] = useState<ExportFolderRecord[]>([]);
  const [exportLoading, setExportLoading] = useState(false);
  const [exportSplitPlans, setExportSplitPlans] = useState<Record<string, ExportSplitPlan>>({});
  const [exportCombinedGroups, setExportCombinedGroups] = useState<ExportGroupRecord[]>([]);
  const [exportCombineSelection, setExportCombineSelection] = useState<string[]>([]);
  const [exportTrainPercent, setExportTrainPercent] = useState(70);
  const [exportValPercent, setExportValPercent] = useState(15);
  const [exportTestPercent, setExportTestPercent] = useState(15);

  const stageRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const renderSeqRef = useRef(0);
  const annotationMaskSeqRef = useRef(0);
  const selectedMaskSeqRef = useRef(0);
  const activeImageIdRef = useRef<number | null>(null);
  const promptPointsRef = useRef<PromptPoint[]>([]);
  const editMaskDataRef = useRef<string | null>(null);
  const panStartRef = useRef<Point | null>(null);
  const loadingReviewCandidateIdsRef = useRef<Set<number>>(new Set());
  const selectedJobRef = useRef<number | null>(null);
  const suppressReviewRenameBlurRef = useRef(false);

  const currentImage = images[currentIndex] ?? null;
  const selectedCandidateObj = candidates.find((candidate) => candidate.localId === selectedCandidate && candidate.imageId === currentImage?.id) ?? null;
  const selectedAnnotationObj = annotations.find((annotation) => annotation.id === selectedAnnotation) ?? null;
  const activeClassName = annotationClassName.trim() || "food";
  const viewportScale = fitScale * zoom;
  const canUndo = undoKinds.length > 0 || candidateHistory.length > 0 || editorHistory.length > 0 || Boolean(selectedAnnotation);
  const canRedo = redoKinds.length > 0 || Boolean(selectedAnnotation);
  const imageFilterKey = useMemo(
    () => JSON.stringify({ q: imageFilenameFilter.trim(), mask: imageMaskFilter, cls: imageClassFilter.trim(), op: imageCountOp, count: imageCountValue.trim() }),
    [imageFilenameFilter, imageMaskFilter, imageClassFilter, imageCountOp, imageCountValue]
  );
  const hasImageFilters = Boolean(imageFilenameFilter.trim() || imageClassFilter.trim() || imageMaskFilter !== "all" || imageCountValue.trim());
  const exportFolderMap = useMemo(() => new Map(exportFolders.map((folder) => [folder.folder, folder])), [exportFolders]);
  const exportBuckets = useMemo(
    () => ({
      train: exportFolders.flatMap((folder) => {
        const plan = exportSplitPlans[folder.folder];
        return plan?.train.length ? [{ folder: folder.folder, split: "train" as const, images: plan.train }] : [];
      }),
      val: exportFolders.flatMap((folder) => {
        const plan = exportSplitPlans[folder.folder];
        return plan?.val.length ? [{ folder: folder.folder, split: "val" as const, images: plan.val }] : [];
      }),
      test: exportFolders.flatMap((folder) => {
        const plan = exportSplitPlans[folder.folder];
        return plan?.test.length ? [{ folder: folder.folder, split: "test" as const, images: plan.test }] : [];
      })
    }),
    [exportFolders, exportSplitPlans]
  );
  const exportAnnotatedCount = useMemo(
    () => exportFolders.reduce((total, folder) => total + folder.annotatedImages.length, 0),
    [exportFolders]
  );

  function clearSelection() {
    setSelectedCandidate(null);
    setSelectedAnnotation(null);
    setEditMaskData(null);
    setResizeDrag(null);
    setEditingTarget(null);
  }

  function unselectOutsideImage(evt: React.MouseEvent) {
    const target = evt.target as HTMLElement;
    if (target.closest("button, input, textarea, select, .candidate, .object, .review-group, .image-item")) return;
    if (target.closest(".image-frame")) return;
    clearSelection();
  }

  function computeFitScale() {
    const stage = stageRef.current;
    if (!stage || !currentImage) return 1;
    const padding = 24;
    return Math.min(
      Math.max((stage.clientWidth - padding) / currentImage.width, 0.05),
      Math.max((stage.clientHeight - padding) / currentImage.height, 0.05)
    );
  }

  function resetViewport(keepUserView = false) {
    if (!currentImage) return;
    const nextFit = computeFitScale();
    setFitScale(nextFit);
    if (!keepUserView || zoom === 1) {
      setZoom(1);
      setPan({ x: 0, y: 0 });
    }
  }

  function selectTool(nextTool: ToolMode) {
    setTool((previous) => (previous === nextTool ? "view" : nextTool));
  }

  function isTypingTarget(target: EventTarget | null) {
    if (!(target instanceof HTMLElement)) return false;
    return Boolean(target.closest("input, textarea, select, [contenteditable='true']"));
  }

  function changeBrushSize(delta: number) {
    setBrushSize((size) => {
      const next = Math.max(2, Math.min(120, size + delta));
      setStatus(`Brush size: ${next}px.`);
      return next;
    });
  }

  function zoomAt(nextZoom: number, clientX?: number, clientY?: number) {
    const stage = stageRef.current;
    if (!stage || !currentImage) return;
    const clampedZoom = Math.max(0.25, Math.min(8, nextZoom));
    const rect = stage.getBoundingClientRect();
    const center = { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
    const cursor = {
      x: clientX ?? center.x,
      y: clientY ?? center.y
    };
    const oldScale = viewportScale || 1;
    const newScale = fitScale * clampedZoom;
    const viewVector = { x: cursor.x - center.x, y: cursor.y - center.y };
    const ratio = newScale / oldScale;
    setPan({
      x: viewVector.x - (viewVector.x - pan.x) * ratio,
      y: viewVector.y - (viewVector.y - pan.y) * ratio
    });
    setZoom(clampedZoom);
  }

  function onWheelStage(evt: React.WheelEvent) {
    if (!currentImage) return;
    evt.preventDefault();
    const factor = evt.deltaY < 0 ? 1.12 : 1 / 1.12;
    zoomAt(zoom * factor, evt.clientX, evt.clientY);
  }

  useEffect(() => {
    if (!currentImage) return;
    activeImageIdRef.current = currentImage.id;
    renderSeqRef.current += 1;
    annotationMaskSeqRef.current += 1;
    selectedMaskSeqRef.current += 1;
    clearOverlayCanvas(currentImage.width, currentImage.height);
    setAnnotations([]);
    setAnnotationMasks({});
    setSelectedAnnotation(null);
    setEditMaskData(null);
    clearMaskCanvas(currentImage.width, currentImage.height);
    resetViewport();
    refreshAnnotations(currentImage.id);
    clearCandidates(true);
  }, [currentImage?.id]);

  useEffect(() => {
    resetViewport();
    const stage = stageRef.current;
    if (!stage) return;
    const observer = new ResizeObserver(() => resetViewport(true));
    observer.observe(stage);
    return () => observer.disconnect();
  }, [currentImage?.id]);

  useEffect(() => {
    drawOverlay();
  }, [candidates, selectedCandidate, editMaskData, annotations, annotationMasks, selectedAnnotation, promptPoints, polygonPoints, showCanvasLabels, currentImage?.id]);

  useEffect(() => {
    promptPointsRef.current = promptPoints;
  }, [promptPoints]);

  useEffect(() => {
    editMaskDataRef.current = editMaskData;
  }, [editMaskData]);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (isTypingTarget(event.target)) return;
      const key = event.key.toLowerCase();
      if ((event.ctrlKey || event.metaKey) && !event.altKey && !event.shiftKey && key === "z") {
        event.preventDefault();
        void undoUniversal();
        return;
      }
      if (key === "n" && !event.ctrlKey && !event.metaKey && !event.altKey) {
        event.preventDefault();
        setTool("point_pos");
        setStatus("Positive point tool selected.");
        return;
      }
      if (key === "b" && !event.ctrlKey && !event.metaKey && !event.altKey) {
        event.preventDefault();
        setTool("brush");
        setStatus("Brush tool selected.");
        return;
      }
      if (key === "v" && !event.ctrlKey && !event.metaKey && !event.altKey) {
        event.preventDefault();
        setTool("erase");
        setStatus("Erase brush tool selected.");
        return;
      }
      if (key === "d" && !event.ctrlKey && !event.metaKey && !event.altKey) {
        event.preventDefault();
        goPreviousImage();
        return;
      }
      if (key === "f" && !event.ctrlKey && !event.metaKey && !event.altKey) {
        event.preventDefault();
        void goNextImage();
        return;
      }
      if ((event.ctrlKey || event.metaKey) && !event.altKey && ["+", "=", "-", "_"].includes(event.key)) {
        event.preventDefault();
        changeBrushSize(event.key === "-" || event.key === "_" ? -3 : 3);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [
    currentIndex,
    images,
    imageHasMore,
    imageLoading,
    undoKinds,
    editorHistory,
    candidateHistory,
    candidates,
    selectedCandidate,
    selectedAnnotation,
    promptPoints,
    polygonPoints,
    currentImage?.id
  ]);

  useEffect(() => {
    loadAnnotationMasks();
  }, [annotations.map((annotation) => `${annotation.id}:${annotation.version}:${annotation.visible}`).join("|"), projectKey]);

  useEffect(() => {
    if (selectedCandidateObj) {
      setSelectedAnnotation(null);
      setEditMaskData(selectedCandidateObj.mask_png);
      hydrateMaskCanvas(selectedCandidateObj.mask_png);
      return;
    }
    loadSelectedAnnotationMask();
  }, [selectedCandidate, selectedAnnotation, currentImage?.id, projectKey]);

  useEffect(() => {
    if (!project) return;
    void refreshClipStatus();
    void refreshBulkJobs();
    void refreshIndexStatus();
  }, [project]);

  useEffect(() => {
    if (rightTab !== "export" || exportFolders.length > 0 || exportLoading) return;
    void loadExportWorkspace();
  }, [rightTab, exportFolders.length, exportLoading]);

  useEffect(() => {
    if (!project) return;
    const timer = window.setInterval(async () => {
      const status = await refreshIndexStatus();
      if (status?.status === "indexing" || status?.status === "completed") {
        await loadImagePage(0, images.length === 0);
      }
      if (status?.status !== "indexing") {
        window.clearInterval(timer);
      }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [project, images.length]);

  useEffect(() => {
    if (!project) return;
    void loadImagePage(0, true);
  }, [project, imageFilterKey]);

  useEffect(() => {
    if (!selectedJobId) return;
    void refreshReviewCandidates(selectedJobId);
    const job = bulkJobs.find((item) => item.id === selectedJobId);
    if (!job || !["queued", "running"].includes(job.status)) return;
    const timer = window.setInterval(async () => {
      const updated = await api.getBulkJob(selectedJobId);
      setBulkJobs((items) => items.map((item) => (item.id === updated.id ? updated : item)));
      await refreshReviewCandidates(selectedJobId);
      if (!["queued", "running"].includes(updated.status)) window.clearInterval(timer);
    }, 1500);
    return () => window.clearInterval(timer);
  }, [selectedJobId, bulkJobs.map((job) => `${job.id}:${job.status}`).join("|")]);

  useEffect(() => {
    if (selectedJobRef.current === selectedJobId) return;
    selectedJobRef.current = selectedJobId;
    loadingReviewCandidateIdsRef.current.clear();
    const selectedReviewCandidate = candidates.find((item) => item.localId === selectedCandidate)?.reviewCandidateId;
    setCandidates((items) => items.filter((item) => !item.reviewCandidateId));
    if (selectedReviewCandidate) {
      setSelectedCandidate(null);
      setEditMaskData(null);
    }
  }, [selectedJobId]);

  useEffect(() => {
    if (!currentImage || !pendingReviewOpen || pendingReviewOpen.candidate.image.id !== currentImage.id) return;
    addReviewCandidateToEditor(pendingReviewOpen);
    setPendingReviewOpen(null);
  }, [currentImage?.id, pendingReviewOpen]);

  useEffect(() => {
    if (!currentImage || reviewCandidates.length === 0) return;
    const imageReviewCandidates = reviewCandidates.filter((candidate) => candidate.status === "pending" && candidate.image.id === currentImage.id);
    if (imageReviewCandidates.length === 0) return;
    void loadReviewCandidatesForImage(currentImage.id, imageReviewCandidates);
  }, [currentImage?.id, reviewCandidates.map((candidate) => `${candidate.id}:${candidate.status}`).join("|")]);

  function resetProjectState() {
    setProject("");
    setProjectKey("");
    setImages([]);
    setImageTotal(0);
    setImageFiltersOpen(false);
    setImageFilenameFilter("");
    setImageMaskFilter("all");
    setImageClassFilter("");
    setImageCountOp("gte");
    setImageCountValue("");
    setImageHasMore(false);
    setIndexStatus(null);
    setCurrentIndex(0);
    setAnnotations([]);
    setAnnotationMasks({});
    setSelectedAnnotation(null);
    setCandidates([]);
    setSelectedCandidate(null);
    setReviewCandidates([]);
    setPendingReviewOpen(null);
    setBulkJobs([]);
    setSelectedJobId(null);
    setClipStatus(null);
    setSearchResults([]);
    setPromptPoints([]);
    setPolygonPoints([]);
    setCandidateHistory([]);
    setCandidateRedo([]);
    setEditorHistory([]);
    setEditorRedo([]);
    setUndoKinds([]);
    setRedoKinds([]);
    setEditMaskData(null);
    setDraftBox(null);
    setBoxStart(null);
    setZoom(1);
    setPan({ x: 0, y: 0 });
    activeImageIdRef.current = null;
    loadingReviewCandidateIdsRef.current.clear();
    clearOverlayCanvas();
    clearMaskCanvas();
  }

  function imageListParams(offset = 0) {
    const parsedCount = imageCountValue.trim() === "" ? null : Number.parseInt(imageCountValue, 10);
    return {
      limit: IMAGE_PAGE_SIZE,
      offset,
      q: imageFilenameFilter.trim() || undefined,
      mask_filter: imageMaskFilter,
      class_name: imageClassFilter.trim() || undefined,
      count_op: Number.isFinite(parsedCount) && parsedCount !== null ? imageCountOp : undefined,
      count_value: Number.isFinite(parsedCount) && parsedCount !== null ? Math.max(0, parsedCount) : null
    };
  }

  function clearImageFilters() {
    setImageFilenameFilter("");
    setImageMaskFilter("all");
    setImageClassFilter("");
    setImageCountOp("gte");
    setImageCountValue("");
  }

  async function openProject() {
    resetProjectState();
    setStatus("Opening project and starting background index...");
    try {
      const summary = await api.openProject(projectPath.trim());
      const nextProjectKey = `${summary.db_path}:${Date.now()}`;
      setProject(summary.root);
      setProjectKey(nextProjectKey);
      const page = await api.listImages({ limit: IMAGE_PAGE_SIZE, offset: 0 });
      setImages(page.items);
      setImageTotal(page.total || summary.image_count);
      setImageHasMore(page.has_more);
      setCurrentIndex(0);
      await refreshIndexStatus();
      setStatus(page.items.length ? `Loaded ${page.items.length}/${page.total} indexed images.` : "Indexing images in the background...");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setStatus(`Open project failed: ${message}`);
      throw error;
    }
  }

  async function loadImagePage(offset = images.length, reset = false) {
    if (imageLoading) return;
    setImageLoading(true);
    try {
      const page = await api.listImages(imageListParams(offset));
      setImageTotal(page.total);
      setImageHasMore(page.has_more);
      setImages((previous) => {
        if (reset) return page.items;
        const byId = new Map(previous.map((image) => [image.id, image]));
        for (const image of page.items) byId.set(image.id, image);
        return Array.from(byId.values()).sort((a, b) => a.file_name.localeCompare(b.file_name));
      });
      if (reset) setCurrentIndex(0);
    } finally {
      setImageLoading(false);
    }
  }

  async function refreshFilteredImageList(focusImageId?: number | null) {
    if (!project) return;
    const previousCurrentId = currentImage?.id ?? null;
    const targetCount = Math.max(IMAGE_PAGE_SIZE, images.length, currentIndex + 1);
    const nextImages: ImageRecord[] = [];
    let offset = 0;
    let total = 0;
    let hasMore = false;

    while (nextImages.length < targetCount) {
      const limit = Math.min(IMAGE_PAGE_SIZE, targetCount - nextImages.length);
      const page = await api.listImages({ ...imageListParams(offset), limit });
      if (offset === 0) total = page.total;
      hasMore = page.has_more;
      nextImages.push(...page.items);
      if (!page.has_more || page.items.length === 0) break;
      offset += page.items.length;
    }

    setImageTotal(total);
    setImageHasMore(hasMore);
    setImages(nextImages);
    setCurrentIndex((previousIndex) => {
      const desiredId = focusImageId ?? previousCurrentId;
      const desiredIndex = desiredId == null ? -1 : nextImages.findIndex((image) => image.id === desiredId);
      if (desiredIndex >= 0) return desiredIndex;
      return Math.min(previousIndex, Math.max(0, nextImages.length - 1));
    });
  }

  async function refreshAfterAnnotationMutation(imageId: number, focusImageId?: number | null) {
    if (activeImageIdRef.current === imageId) await refreshAnnotations(imageId);
    await refreshFilteredImageList(focusImageId);
  }

  async function refreshIndexStatus() {
    try {
      const status = await api.projectIndexStatus();
      setIndexStatus(status);
      return status;
    } catch {
      setIndexStatus(null);
      return null;
    }
  }

  function navigateToImage(image: ImageRecord) {
    const existing = images.findIndex((item) => item.id === image.id);
    if (existing >= 0) {
      setCurrentIndex(existing);
      return;
    }
    const next = [...images.filter((item) => item.id !== image.id), image].sort((a, b) => a.file_name.localeCompare(b.file_name));
    setImages(next);
    setCurrentIndex(next.findIndex((item) => item.id === image.id));
  }

  async function goNextImage() {
    if (currentIndex >= images.length - 1 && imageHasMore) {
      await loadImagePage(images.length, false);
      setCurrentIndex(currentIndex + 1);
      return;
    }
    setCurrentIndex(Math.min(images.length - 1, currentIndex + 1));
  }

  function goPreviousImage() {
    setCurrentIndex(Math.max(0, currentIndex - 1));
  }

  function onImageListScroll(evt: React.UIEvent<HTMLElement>) {
    const target = evt.currentTarget;
    if (!imageHasMore || imageLoading) return;
    if (target.scrollTop + target.clientHeight >= target.scrollHeight - 240) {
      void loadImagePage(images.length, false);
    }
  }

  async function refreshAnnotations(imageId: number) {
    const loaded = await api.listAnnotations(imageId);
    if (activeImageIdRef.current !== imageId) return;
    setAnnotations(loaded);
    setSelectedAnnotation((previous) => (previous && loaded.some((item) => item.id === previous) ? previous : loaded[0]?.id ?? null));
    setAnnotationMasks((previous) => {
      const visibleIds = new Set(loaded.filter((item) => item.visible).map((item) => item.id));
      return Object.fromEntries(Object.entries(previous).filter(([id]) => visibleIds.has(Number(id))));
    });
    if (selectedAnnotation && !loaded.some((item) => item.id === selectedAnnotation)) {
      setEditMaskData(null);
    }
  }

  async function loadAnnotationMasks() {
    const seq = ++annotationMaskSeqRef.current;
    const imageId = currentImage?.id ?? null;
    const next: Record<number, string> = {};
    await Promise.all(
      annotations
        .filter((annotation) => annotation.visible)
        .map(async (annotation) => {
          try {
            const response = await fetch(annotationMaskUrl(annotation.id, projectKey));
            const blob = await response.blob();
            next[annotation.id] = await blobToDataUrl(blob);
          } catch {
            // A missing mask should not break annotation display.
          }
        })
    );
    if (seq !== annotationMaskSeqRef.current || activeImageIdRef.current !== imageId) return;
    setAnnotationMasks(next);
  }

  function pushCandidateHistory() {
    setCandidateHistory((history) => [...history, snapshotWorkState()]);
    setCandidateRedo([]);
    setEditorRedo([]);
    setUndoKinds((items) => [...items, "candidate"]);
    setRedoKinds([]);
  }

  function snapshotWorkState(): WorkState {
    return {
      candidates,
      selectedCandidate,
      promptPoints,
      polygonPoints
    };
  }

  function restoreWorkState(state: WorkState) {
    setCandidates(state.candidates);
    setSelectedCandidate(state.selectedCandidate);
    setPromptPoints(state.promptPoints);
    setPolygonPoints(state.polygonPoints);
  }

  function recordEditorAction(action: EditorAction) {
    setEditorHistory((history) => [...history, action]);
    setEditorRedo([]);
    setCandidateRedo([]);
    setUndoKinds((items) => [...items, "editor"]);
    setRedoKinds([]);
  }

  async function deleteAnnotationLocal(id: number) {
    setAnnotationMasks((previous) => {
      const next = { ...previous };
      delete next[id];
      return next;
    });
    setAnnotations((items) => items.filter((item) => item.id !== id));
    if (selectedAnnotation === id) {
      setSelectedAnnotation(null);
      setEditMaskData(null);
    }
    await api.deleteAnnotation(id);
  }

  async function restoreAnnotationFromSnapshot(snapshot: AnnotationRecord, maskPng: string) {
    const createStatus = snapshot.status === "pending" ? "pending" : "accepted";
    const restored = await api.createAnnotation(
      snapshot.image_id,
      snapshot.category_name ?? "food",
      maskPng,
      createStatus,
      snapshot.score
    );
    if (snapshot.status === "rejected") await api.updateAnnotation(restored.id, { status: "rejected" });
    if (!snapshot.visible) await api.updateAnnotation(restored.id, { visible: false });
    return { ...restored, visible: snapshot.visible };
  }

  function setNewCandidates(raw: MaskCandidate[], source: string) {
    const next = raw.map((candidate, index) => ({
      ...candidate,
      localId: Date.now() + index,
      imageId: currentImage?.id ?? 0,
      name: activeClassName,
      visible: true
    }));
    setCandidates(next);
    setSelectedCandidate(next[0]?.localId ?? null);
    setSelectedAnnotation(null);
  }

  function filterPromptCandidates(raw: MaskCandidate[]) {
    const sorted = [...raw].sort((a, b) => (b.score ?? -Infinity) - (a.score ?? -Infinity));
    const thresholded =
      promptFilterMode === "threshold_top_k"
        ? sorted.filter((candidate) => candidate.score == null || candidate.score >= promptThreshold)
        : sorted;
    return thresholded.slice(0, Math.max(1, promptTopK));
  }

  function clearCandidates(skipConfirm = false) {
    if (!skipConfirm && candidates.length > 0 && !window.confirm("Clear unsaved SAM/manual candidates? Saved objects will stay.")) return;
    const hadCandidates = candidates.length > 0;
    setCandidates([]);
    setSelectedCandidate(null);
    setPromptPoints([]);
    setPolygonPoints([]);
    setCandidateHistory([]);
    setCandidateRedo([]);
    setEditMaskData(null);
    if (hadCandidates) setStatus("Cleared unsaved candidates. Saved objects were not changed.");
  }

  async function loadSelectedAnnotationMask() {
    const annotationId = selectedAnnotation;
    const imageId = currentImage?.id ?? null;
    const seq = ++selectedMaskSeqRef.current;
    if (!annotationId) {
      setEditMaskData(null);
      return;
    }
    try {
      const response = await fetch(annotationMaskUrl(annotationId, projectKey));
      const blob = await response.blob();
      const dataUrl = await blobToDataUrl(blob);
      if (seq !== selectedMaskSeqRef.current || activeImageIdRef.current !== imageId || selectedAnnotation !== annotationId) return;
      setEditMaskData(dataUrl);
      hydrateMaskCanvas(dataUrl);
    } catch {
      setEditMaskData(null);
    }
  }

  function clearOverlayCanvas(width?: number, height?: number) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (width && height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext("2d");
    ctx?.clearRect(0, 0, canvas.width, canvas.height);
  }

  function clearMaskCanvas(width?: number, height?: number) {
    const canvas = maskCanvasRef.current;
    if (!canvas) return;
    if (width && height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext("2d");
    ctx?.clearRect(0, 0, canvas.width, canvas.height);
  }

  function canvasPoint(evt: React.MouseEvent) {
    const img = imageRef.current;
    if (!img || !currentImage) return null;
    const rect = img.getBoundingClientRect();
    if (evt.clientX < rect.left || evt.clientX > rect.right || evt.clientY < rect.top || evt.clientY > rect.bottom) {
      return null;
    }
    const x = ((evt.clientX - rect.left) / rect.width) * currentImage.width;
    const y = ((evt.clientY - rect.top) / rect.height) * currentImage.height;
    return { x: Math.max(0, Math.min(currentImage.width, x)), y: Math.max(0, Math.min(currentImage.height, y)) };
  }

  async function runPoint(point: Point, label: 0 | 1) {
    if (!currentImage) return;
    const nextPoints = [...promptPoints, { ...point, label }];
    pushCandidateHistory();
    await runPointSet(nextPoints);
  }

  async function runPointSet(nextPoints: PromptPoint[]) {
    if (!currentImage) return;
    setPromptPoints(nextPoints);
    setStatus("Running SAM with point set...");
    const res = await api.samPrompt({
      image_id: currentImage.id,
      prompt_type: "point",
      points: nextPoints,
      category_name: activeClassName,
      accept: false
    });
    const filtered = filterPromptCandidates(res.candidates);
    setNewCandidates(filtered, "point");
    setStatus(res.sam_error ? `SAM failed: ${res.sam_error}` : res.candidates.length ? `SAM returned ${filtered.length}/${res.candidates.length} candidate masks after filters.` : "SAM returned no masks.");
  }

  async function runTextPrompt() {
    if (!currentImage) return;
    setStatus("Running SAM text prompt...");
    pushCandidateHistory();
    const res = await api.samPrompt({
      image_id: currentImage.id,
      prompt_type: "text",
      text: promptText,
      category_name: activeClassName,
      accept: false
    });
    setPromptPoints([]);
    const filtered = filterPromptCandidates(res.candidates);
    setNewCandidates(filtered, "text");
    setStatus(res.sam_error ? `SAM failed: ${res.sam_error}` : res.candidates.length ? `SAM returned ${filtered.length}/${res.candidates.length} candidate masks after filters.` : "SAM returned no masks.");
  }

  async function runBoxPrompt(box: { x: number; y: number; width: number; height: number }) {
    if (!currentImage) return;
    setStatus("Running SAM box prompt...");
    pushCandidateHistory();
    const res = await api.samPrompt({
      image_id: currentImage.id,
      prompt_type: "box",
      boxes: [{ ...box, label: 1 }],
      category_name: activeClassName,
      accept: false
    });
    setPromptPoints([]);
    const filtered = filterPromptCandidates(res.candidates);
    setNewCandidates(filtered, "box");
    setStatus(res.sam_error ? `SAM failed: ${res.sam_error}` : res.candidates.length ? `SAM returned ${filtered.length}/${res.candidates.length} candidate masks after filters.` : "SAM returned no masks.");
  }

  async function acceptSelectedCandidate() {
    if (!selectedCandidateObj || !currentImage) return;
    const imageId = currentImage.id;
    const acceptedCandidate = selectedCandidateObj;
    const previousCandidates = candidates;
    const previousSelectedCandidate = selectedCandidate;
    const previousSelectedAnnotation = selectedAnnotation;
    const created = await api.createAnnotation(imageId, selectedCandidateObj.name || activeClassName, selectedCandidateObj.mask_png, "accepted");
    let activeAnnotationId = created.id;
    if (selectedCandidateObj.reviewCandidateId) {
      await api.linkAcceptedReviewCandidate(selectedCandidateObj.reviewCandidateId, created.id);
      setReviewCandidates((items) => items.filter((item) => item.id !== selectedCandidateObj.reviewCandidateId));
    }
    setSelectedAnnotation(created.id);
    setAnnotationMasks((previous) => ({ ...previous, [created.id]: selectedCandidateObj.mask_png }));
    setAnnotations((items) => [...items.filter((item) => item.id !== created.id), { ...created, visible: true }]);
    setCandidates((items) => items.filter((item) => item.localId !== selectedCandidateObj.localId));
    setSelectedCandidate(null);
    await refreshAfterAnnotationMutation(imageId);
    recordEditorAction({
      label: "accept selected candidate",
      undo: async () => {
        await deleteAnnotationLocal(activeAnnotationId);
        if (acceptedCandidate.reviewCandidateId) {
          const reopened = await api.reopenReviewCandidate(acceptedCandidate.reviewCandidateId);
          setReviewCandidates((items) => items.some((item) => item.id === reopened.id) ? items : [reopened, ...items]);
        }
        setCandidates(previousCandidates);
        setSelectedCandidate(previousSelectedCandidate);
        setSelectedAnnotation(previousSelectedAnnotation);
        await refreshAfterAnnotationMutation(imageId, imageId);
      },
      redo: async () => {
        const redone = await api.createAnnotation(imageId, acceptedCandidate.name || activeClassName, acceptedCandidate.mask_png, "accepted", acceptedCandidate.score);
        activeAnnotationId = redone.id;
        if (acceptedCandidate.reviewCandidateId) {
          await api.linkAcceptedReviewCandidate(acceptedCandidate.reviewCandidateId, redone.id);
          setReviewCandidates((items) => items.filter((item) => item.id !== acceptedCandidate.reviewCandidateId));
        }
        setCandidates((items) => items.filter((item) => item.localId !== acceptedCandidate.localId));
        setSelectedCandidate(null);
        setSelectedAnnotation(redone.id);
        setAnnotationMasks((previous) => ({ ...previous, [redone.id]: acceptedCandidate.mask_png }));
        await refreshAfterAnnotationMutation(imageId);
      }
    });
    setStatus(`Saved candidate ${candidateIndex(selectedCandidateObj)} as object #${created.id}.`);
  }

  async function acceptAllCandidates() {
    if (!currentImage || candidates.length === 0) return;
    const imageId = currentImage.id;
    const acceptedCandidates = candidates;
    const previousSelectedCandidate = selectedCandidate;
    const createdAnnotations: AnnotationRecord[] = [];
    const createdMasks: Record<number, string> = {};
    let activeAnnotationIds: number[] = [];
    for (const candidate of acceptedCandidates) {
      const created = await api.createAnnotation(imageId, candidate.name || activeClassName, candidate.mask_png, "accepted", candidate.score);
      if (candidate.reviewCandidateId) {
        await api.linkAcceptedReviewCandidate(candidate.reviewCandidateId, created.id);
      }
      createdAnnotations.push({ ...created, visible: true });
      createdMasks[created.id] = candidate.mask_png;
      activeAnnotationIds.push(created.id);
    }
    setReviewCandidates((items) => items.filter((item) => !acceptedCandidates.some((candidate) => candidate.reviewCandidateId === item.id)));
    setAnnotations((items) => [...items, ...createdAnnotations]);
    setAnnotationMasks((previous) => ({ ...previous, ...createdMasks }));
    clearCandidates(true);
    await refreshAfterAnnotationMutation(imageId);
    recordEditorAction({
      label: "accept all candidates",
      undo: async () => {
        await Promise.all(activeAnnotationIds.map((id) => deleteAnnotationLocal(id)));
        for (const candidate of acceptedCandidates) {
          if (candidate.reviewCandidateId) {
            const reopened = await api.reopenReviewCandidate(candidate.reviewCandidateId);
            setReviewCandidates((items) => items.some((item) => item.id === reopened.id) ? items : [reopened, ...items]);
          }
        }
        setCandidates(acceptedCandidates);
        setSelectedCandidate(previousSelectedCandidate);
        await refreshAfterAnnotationMutation(imageId, imageId);
      },
      redo: async () => {
        const nextIds: number[] = [];
        const nextAnnotations: AnnotationRecord[] = [];
        const nextMasks: Record<number, string> = {};
        for (const candidate of acceptedCandidates) {
          const created = await api.createAnnotation(imageId, candidate.name || activeClassName, candidate.mask_png, "accepted", candidate.score);
          if (candidate.reviewCandidateId) await api.linkAcceptedReviewCandidate(candidate.reviewCandidateId, created.id);
          nextIds.push(created.id);
          nextAnnotations.push({ ...created, visible: true });
          nextMasks[created.id] = candidate.mask_png;
        }
        activeAnnotationIds = nextIds;
        setReviewCandidates((items) => items.filter((item) => !acceptedCandidates.some((candidate) => candidate.reviewCandidateId === item.id)));
        setAnnotations((items) => [...items, ...nextAnnotations]);
        setAnnotationMasks((previous) => ({ ...previous, ...nextMasks }));
        setCandidates([]);
        setSelectedCandidate(null);
        await refreshAfterAnnotationMutation(imageId);
      }
    });
    setStatus("Saved all candidate masks as objects.");
  }

  async function rejectPending(id: number) {
    const previous = annotations.find((annotation) => annotation.id === id);
    const imageId = previous?.image_id ?? currentImage?.id;
    await api.updateAnnotation(id, { status: "rejected" });
    if (imageId) await refreshAfterAnnotationMutation(imageId);
    if (previous) {
      recordEditorAction({
        label: `reject object #${id}`,
        undo: async () => {
          await api.updateAnnotation(id, { status: previous.status as "accepted" | "pending" | "rejected" });
          if (imageId) await refreshAfterAnnotationMutation(imageId, imageId);
        },
        redo: async () => {
          await api.updateAnnotation(id, { status: "rejected" });
          if (imageId) await refreshAfterAnnotationMutation(imageId);
        }
      });
    }
  }

  async function acceptPending(id: number) {
    const previous = annotations.find((annotation) => annotation.id === id);
    const imageId = previous?.image_id ?? currentImage?.id;
    await api.updateAnnotation(id, { status: "accepted" });
    if (imageId) await refreshAfterAnnotationMutation(imageId);
    if (previous) {
      recordEditorAction({
        label: `accept object #${id}`,
        undo: async () => {
          await api.updateAnnotation(id, { status: previous.status as "accepted" | "pending" | "rejected" });
          if (imageId) await refreshAfterAnnotationMutation(imageId, imageId);
        },
        redo: async () => {
          await api.updateAnnotation(id, { status: "accepted" });
          if (imageId) await refreshAfterAnnotationMutation(imageId);
        }
      });
    }
  }

  async function removeAnnotation(id: number) {
    const removed = annotations.find((annotation) => annotation.id === id);
    let removedMask: string | null = annotationMasks[id] ?? null;
    if (removed && !removedMask) {
      try {
        const response = await fetch(annotationMaskUrl(id, projectKey));
        removedMask = await blobToDataUrl(await response.blob());
      } catch {
        removedMask = null;
      }
    }
    await deleteAnnotationLocal(id);
    if (removed?.image_id) await refreshAfterAnnotationMutation(removed.image_id, removed.image_id);
    if (removed && removedMask) {
      let activeAnnotationId: number | null = null;
      recordEditorAction({
        label: `delete object #${id}`,
        undo: async () => {
          const restored = await restoreAnnotationFromSnapshot(removed, removedMask);
          activeAnnotationId = restored.id;
          setAnnotations((items) => [...items.filter((item) => item.id !== restored.id), restored]);
          if (restored.visible) setAnnotationMasks((previous) => ({ ...previous, [restored.id]: removedMask }));
          setSelectedAnnotation(restored.id);
          setSelectedCandidate(null);
          await refreshAfterAnnotationMutation(removed.image_id, removed.image_id);
        },
        redo: async () => {
          if (activeAnnotationId) {
            await deleteAnnotationLocal(activeAnnotationId);
            activeAnnotationId = null;
          }
          await refreshAfterAnnotationMutation(removed.image_id);
        }
      });
    }
    setStatus(`Deleted saved object #${id}. Unsaved candidates remain until cleared.`);
  }

  async function toggleAnnotationVisibility(annotation: AnnotationRecord) {
    const nextVisible = !annotation.visible;
    const previousVisible = annotation.visible;
    setAnnotations((items) => items.map((item) => (item.id === annotation.id ? { ...item, visible: nextVisible } : item)));
    if (!nextVisible) {
      setAnnotationMasks((previous) => {
        const next = { ...previous };
        delete next[annotation.id];
        return next;
      });
      if (selectedAnnotation === annotation.id) setEditMaskData(null);
    }
    await api.updateAnnotation(annotation.id, { visible: nextVisible });
    if (currentImage) await refreshAnnotations(currentImage.id);
    recordEditorAction({
      label: `${nextVisible ? "show" : "hide"} object #${annotation.id}`,
      undo: async () => {
        await api.updateAnnotation(annotation.id, { visible: previousVisible });
        if (currentImage) await refreshAnnotations(currentImage.id);
      },
      redo: async () => {
        await api.updateAnnotation(annotation.id, { visible: nextVisible });
        if (currentImage) await refreshAnnotations(currentImage.id);
      }
    });
    setStatus(`${nextVisible ? "Showing" : "Hiding"} saved object #${annotation.id}.`);
  }

  function toggleCandidateVisibility(localId: number) {
    setCandidates((items) => items.map((item) => (item.localId === localId ? { ...item, visible: !item.visible } : item)));
  }

  async function showAllObjects() {
    if (!currentImage || annotations.length === 0) return;
    setStatus("Showing all saved object masks...");
    setAnnotations((items) => items.map((item) => ({ ...item, visible: true })));
    await Promise.all(
      annotations
        .filter((annotation) => !annotation.visible)
        .map((annotation) => api.updateAnnotation(annotation.id, { visible: true }))
    );
    await refreshAnnotations(currentImage.id);
    setStatus("All saved object masks are visible.");
  }

  async function loadAnnotationMaskData(annotation: AnnotationRecord) {
    const existing = annotationMasks[annotation.id];
    if (existing) return existing;
    const response = await fetch(annotationMaskUrl(annotation.id, projectKey));
    return blobToDataUrl(await response.blob());
  }

  async function unionAnnotationMasks(items: Array<{ annotation: AnnotationRecord; maskPng: string }>) {
    if (!currentImage) return null;
    const canvas = document.createElement("canvas");
    canvas.width = currentImage.width;
    canvas.height = currentImage.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    const output = ctx.createImageData(currentImage.width, currentImage.height);
    for (const item of items) {
      const img = await loadImage(item.maskPng);
      const pixels = maskPixels(img, currentImage.width, currentImage.height);
      if (!pixels) continue;
      for (let i = 0; i < pixels.data.length; i += 4) {
        if (!isMaskPixel(pixels.data, i)) continue;
        output.data[i] = 255;
        output.data[i + 1] = 255;
        output.data[i + 2] = 255;
        output.data[i + 3] = 255;
      }
    }
    ctx.putImageData(output, 0, 0);
    return canvas.toDataURL("image/png");
  }

  async function combineSavedObjectMasks() {
    if (!currentImage) return;
    const imageId = currentImage.id;
    const combinable = annotations.filter((annotation) => annotation.image_id === imageId && annotation.status === "accepted");
    if (combinable.length < 2) {
      setStatus("Need at least two saved objects to combine.");
      return;
    }
    const label = activeClassName || combinable[0].category_name || "food";
    if (!window.confirm(`Combine ${combinable.length} saved objects into one "${label}" mask? The original saved objects will be removed.`)) {
      return;
    }
    setStatus(`Combining ${combinable.length} saved object masks...`);
    try {
      const snapshots = await Promise.all(
        combinable.map(async (annotation) => ({
          annotation,
          maskPng: await loadAnnotationMaskData(annotation)
        }))
      );
      const combinedMask = await unionAnnotationMasks(snapshots);
      if (!combinedMask) {
        setStatus("Could not combine saved object masks.");
        return;
      }
      const created = await api.createAnnotation(imageId, label, combinedMask, "accepted");
      await Promise.all(combinable.map((annotation) => deleteAnnotationLocal(annotation.id)));
      setSelectedAnnotation(created.id);
      setSelectedCandidate(null);
      setAnnotationMasks((previous) => {
        const next = { ...previous, [created.id]: combinedMask };
        combinable.forEach((annotation) => delete next[annotation.id]);
        return next;
      });
      await refreshAfterAnnotationMutation(imageId, imageId);
      let activeCombinedId: number | null = created.id;
      let activeRestoredIds: number[] = [];
      recordEditorAction({
        label: `combine ${combinable.length} objects`,
        undo: async () => {
          if (activeCombinedId) {
            await deleteAnnotationLocal(activeCombinedId);
            activeCombinedId = null;
          }
          const restored = await Promise.all(
            snapshots.map((snapshot) => restoreAnnotationFromSnapshot(snapshot.annotation, snapshot.maskPng))
          );
          activeRestoredIds = restored.map((annotation) => annotation.id);
          setAnnotationMasks((previous) => {
            const next = { ...previous };
            restored.forEach((annotation, index) => {
              if (annotation.visible) next[annotation.id] = snapshots[index].maskPng;
            });
            return next;
          });
          setSelectedAnnotation(restored[0]?.id ?? null);
          await refreshAfterAnnotationMutation(imageId, imageId);
        },
        redo: async () => {
          const redone = await api.createAnnotation(imageId, label, combinedMask, "accepted");
          activeCombinedId = redone.id;
          await Promise.all(activeRestoredIds.map((id) => deleteAnnotationLocal(id).catch(() => undefined)));
          activeRestoredIds = [];
          setSelectedAnnotation(redone.id);
          setAnnotationMasks((previous) => ({ ...previous, [redone.id]: combinedMask }));
          await refreshAfterAnnotationMutation(imageId, imageId);
        }
      });
      setStatus(`Combined ${combinable.length} saved objects into object #${created.id}.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Combine saved object masks failed.");
    }
  }

  function renameCandidate(localId: number, value: string) {
    const name = value.trim() || "food";
    setCandidates((items) => items.map((candidate) => (candidate.localId === localId ? { ...candidate, name } : candidate)));
  }

  async function renameAnnotation(annotation: AnnotationRecord, value: string) {
    const name = value.trim() || "food";
    setAnnotations((items) => items.map((item) => (item.id === annotation.id ? { ...item, category_name: name } : item)));
    await api.updateAnnotation(annotation.id, { category_name: name });
    await refreshAfterAnnotationMutation(annotation.image_id);
    setStatus(`Renamed object #${annotation.id} to ${name}.`);
  }

  async function renameReviewCandidate(candidate: ReviewCandidateRecord, value: string) {
    const name = value.trim() || "food";
    if (name === candidate.category_name) return;
    const updated = await api.updateReviewCandidate(candidate.id, name);
    setReviewCandidates((items) => items.map((item) => (item.id === updated.id ? updated : item)));
    setCandidates((items) =>
      items.map((item) => (item.reviewCandidateId === updated.id ? { ...item, name: updated.category_name } : item))
    );
    setStatus(`Renamed review candidate #${candidate.id} to ${name}.`);
  }

  async function renameAllPendingReviewCandidates() {
    const name = bulkRenameText.trim();
    if (!name || reviewCandidates.length === 0 || selectedJobId == null) return;
    suppressReviewRenameBlurRef.current = true;
    setStatus(`Renaming ${reviewCandidates.length} pending review candidates...`);
    try {
      const updated = await api.renamePendingReviewCandidates(selectedJobId, name);
      const updatedIds = new Set(updated.ids);
      setReviewCandidates((items) =>
        items.map((item) => (updatedIds.has(item.id) ? { ...item, category_name: updated.category_name } : item))
      );
      setCandidates((items) =>
        items.map((item) =>
          item.reviewCandidateId && updatedIds.has(item.reviewCandidateId)
            ? { ...item, name: updated.category_name }
            : item
        )
      );
      await refreshReviewCandidates(selectedJobId);
      setStatus(`Renamed ${updated.updated} pending review candidates to ${updated.category_name}.`);
    } finally {
      window.setTimeout(() => {
        suppressReviewRenameBlurRef.current = false;
      }, 0);
    }
  }

  async function acceptAllPendingReviewCandidates() {
    if (selectedJobId == null || reviewCandidates.length === 0) return;
    const count = reviewCandidates.length;
    if (count > 25 && !window.confirm(`Accept all ${count} pending candidates for Job #${selectedJobId}? This will create saved annotations.`)) {
      return;
    }
    try {
      setStatus(`Accepting ${count} pending review candidates...`);
      const accepted = await api.acceptPendingReviewCandidates(selectedJobId);
      const acceptedIds = new Set(accepted.ids);
      setReviewCandidates((items) => items.filter((item) => !acceptedIds.has(item.id)));
      setCandidates((items) => items.filter((item) => !item.reviewCandidateId || !acceptedIds.has(item.reviewCandidateId)));
      if (selectedCandidateObj?.reviewCandidateId && acceptedIds.has(selectedCandidateObj.reviewCandidateId)) {
        setSelectedCandidate(null);
        setEditMaskData(null);
      }
      if (currentImage) await refreshAnnotations(currentImage.id);
      await refreshFilteredImageList();
      await refreshReviewCandidates(selectedJobId);
      setStatus(`Accepted ${accepted.accepted} pending review candidates from Job #${selectedJobId}.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Accept all pending review candidates failed.");
    }
  }

  async function undoUniversal() {
    const nextKind = undoKinds[undoKinds.length - 1];
    if (nextKind === "editor" && editorHistory.length > 0) {
      const action = editorHistory[editorHistory.length - 1];
      await action.undo();
      setEditorRedo((redo) => [action, ...redo]);
      setEditorHistory((history) => history.slice(0, -1));
      setUndoKinds((items) => items.slice(0, -1));
      setRedoKinds((items) => ["editor", ...items]);
      setStatus(`Undid ${action.label}.`);
      return;
    }
    if (nextKind === "candidate" && candidateHistory.length > 0) {
      const previous = candidateHistory[candidateHistory.length - 1];
      setCandidateRedo((redo) => [snapshotWorkState(), ...redo]);
      setCandidateHistory((history) => history.slice(0, -1));
      setUndoKinds((items) => items.slice(0, -1));
      setRedoKinds((items) => ["candidate", ...items]);
      restoreWorkState(previous);
      setStatus("Undid candidate/manual edit.");
      return;
    }
    if (editorHistory.length > 0) {
      const action = editorHistory[editorHistory.length - 1];
      await action.undo();
      setEditorRedo((redo) => [action, ...redo]);
      setEditorHistory((history) => history.slice(0, -1));
      setRedoKinds((items) => ["editor", ...items]);
      setStatus(`Undid ${action.label}.`);
      return;
    }
    if (candidateHistory.length > 0) {
      const previous = candidateHistory[candidateHistory.length - 1];
      setCandidateRedo((redo) => [snapshotWorkState(), ...redo]);
      setCandidateHistory((history) => history.slice(0, -1));
      setRedoKinds((items) => ["candidate", ...items]);
      restoreWorkState(previous);
      setStatus("Undid candidate/manual edit.");
      return;
    }
    if (!selectedAnnotation || !currentImage) return;
    await api.undoAnnotation(selectedAnnotation);
    await refreshAnnotations(currentImage.id);
    await loadSelectedAnnotationMask();
    setStatus("Undid saved object mask edit.");
  }

  async function redoUniversal() {
    const nextKind = redoKinds[0];
    if (nextKind === "candidate" && candidateRedo.length > 0) {
      const next = candidateRedo[0];
      setCandidateHistory((history) => [...history, snapshotWorkState()]);
      setCandidateRedo((redo) => redo.slice(1));
      setRedoKinds((items) => items.slice(1));
      setUndoKinds((items) => [...items, "candidate"]);
      restoreWorkState(next);
      setStatus("Redid candidate/manual edit.");
      return;
    }
    if (nextKind === "editor" && editorRedo.length > 0) {
      const action = editorRedo[0];
      if (!action.redo) return;
      await action.redo();
      setEditorHistory((history) => [...history, action]);
      setEditorRedo((redo) => redo.slice(1));
      setRedoKinds((items) => items.slice(1));
      setUndoKinds((items) => [...items, "editor"]);
      setStatus(`Redid ${action.label}.`);
      return;
    }
    if (!selectedAnnotation || !currentImage) return;
    await api.redoAnnotation(selectedAnnotation);
    await refreshAnnotations(currentImage.id);
    await loadSelectedAnnotationMask();
    setStatus("Redid saved object mask edit.");
  }

  async function runSearch() {
    try {
      setStatus("Searching CLIP embeddings...");
      const results = await api.searchClip({ text: searchText, limit: 50 });
      setSearchResults(results);
      setStatus(`Found ${results.length} similar images.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "CLIP search failed.");
      await refreshClipStatus();
    }
  }

  async function runCurrentImageSearch() {
    if (!currentImage) return;
    try {
      setStatus("Searching images similar to current image...");
      const results = await api.searchClip({ image_id: currentImage.id, limit: 50 });
      setSearchResults(results);
      setStatus(`Found ${results.length} similar images.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "CLIP image search failed.");
      await refreshClipStatus();
    }
  }

  async function refreshClipStatus() {
    try {
      setClipStatus(await api.clipStatus());
    } catch {
      setClipStatus(null);
    }
  }

  async function runClipIndex(force = false) {
    try {
      setStatus(force ? "Rebuilding CLIP index..." : "Indexing missing CLIP embeddings...");
      const result = force ? await api.rebuildClip() : await api.indexClip();
      setClipStatus(result);
      setStatus(`CLIP index ready: ${result.indexed}/${result.total} newly indexed.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "CLIP indexing failed.");
      await refreshClipStatus();
    }
  }

  async function refreshBulkJobs() {
    try {
      const jobs = await api.listBulkJobs();
      setBulkJobs(jobs);
      setSelectedJobId((previous) => previous ?? jobs[0]?.id ?? null);
    } catch {
      setBulkJobs([]);
    }
  }

  async function refreshReviewCandidates(jobId: number) {
    const candidates = await api.listReviewCandidates(jobId, "pending");
    setReviewCandidates(candidates);
  }

  function removeBulkJobsFromUi(ids: number[]) {
    const removed = new Set(ids);
    const nextJobs = bulkJobs.filter((job) => !removed.has(job.id));
    setBulkJobs(nextJobs);
    if (selectedJobId && removed.has(selectedJobId)) {
      setSelectedJobId(nextJobs[0]?.id ?? null);
      setReviewCandidates([]);
      setCandidates((items) => items.filter((item) => !item.reviewCandidateId));
      setSelectedCandidate((previous) => {
        const selected = candidates.find((item) => item.localId === previous);
        return selected?.reviewCandidateId ? null : previous;
      });
    }
  }

  async function deleteBulkJob(job: BulkJobRecord) {
    if (["queued", "running"].includes(job.status)) {
      setStatus("Queued or running jobs cannot be removed yet.");
      return;
    }
    if (!window.confirm(`Remove Job #${job.id} from bulk history? Accepted saved objects will stay.`)) return;
    try {
      await api.deleteBulkJob(job.id);
      removeBulkJobsFromUi([job.id]);
      setStatus(`Removed bulk job #${job.id}. Saved objects were not changed.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Could not remove bulk job.");
    }
  }

  async function cancelBulkJob(job: BulkJobRecord) {
    if (!["queued", "running"].includes(job.status)) return;
    if (!window.confirm(`Cancel Job #${job.id}? Candidates already created will stay pending for review.`)) return;
    try {
      const updated = await api.cancelBulkJob(job.id);
      setBulkJobs((items) => items.map((item) => (item.id === updated.id ? updated : item)));
      setStatus(updated.status === "cancelled" ? `Cancelled bulk job #${job.id}.` : `Cancel requested for bulk job #${job.id}.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Could not cancel bulk job.");
    }
  }

  async function clearFinishedBulkJobs() {
    const removable = bulkJobs.filter((job) => ["completed", "failed", "cancelled"].includes(job.status));
    if (removable.length === 0) {
      setStatus("There are no completed or failed bulk jobs to remove.");
      return;
    }
    if (!window.confirm(`Remove ${removable.length} completed/failed bulk jobs from history? Accepted saved objects will stay.`)) return;
    try {
      const result = await api.deleteFinishedBulkJobs();
      removeBulkJobsFromUi(result.deleted);
      setStatus(`Removed ${result.deleted.length} finished bulk jobs. Saved objects were not changed.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Could not clear finished bulk jobs.");
    }
  }

  async function runBulk() {
    const totalImages = imageTotal || images.length;
    if (bulkMode === "all" && totalImages > 200 && !window.confirm(`Run bulk SAM on all ${totalImages} images? This can take a while.`)) {
      return;
    }
    if (bulkMode === "clip_filtered" && searchResults.length === 0) {
      setStatus("Run CLIP search first or switch bulk mode to All Images.");
      return;
    }
    try {
      setStatus("Starting bulk review job...");
      const job = await api.createBulkJob({
        mode: bulkMode,
        text: bulkPrompt,
        category_name: bulkPrompt,
        confidence_threshold: bulkThreshold,
        top_k: bulkTopK,
        max_images: bulkMode === "clip_filtered" ? bulkMaxImages : undefined,
        clip_image_ids: bulkMode === "clip_filtered" ? searchResults.slice(0, bulkMaxImages).map((result) => result.image.id) : []
      });
      setRightTab("bulk");
      setBulkJobs((items) => [job, ...items.filter((item) => item.id !== job.id)]);
      setSelectedJobId(job.id);
      setReviewCandidates([]);
      setStatus(`Started bulk job #${job.id}.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Bulk job failed to start.");
    }
  }

  async function runPropagate() {
    if (!selectedAnnotation) return;
    setStatus("Propagating to similar images...");
    const result = await api.propagate(selectedAnnotation);
    setStatus(`Created ${result.annotations.length} propagated pending masks.`);
  }

  async function acceptReviewCandidate(candidateId: number) {
    const openedCandidate = candidates.find((item) => item.reviewCandidateId === candidateId);
    const accepted = await api.acceptReviewCandidate(candidateId);
    setReviewCandidates((items) => items.filter((item) => item.id !== candidateId));
    setCandidates((items) => items.filter((item) => item.reviewCandidateId !== candidateId));
    if (selectedCandidateObj?.reviewCandidateId === candidateId) setSelectedCandidate(null);
    await refreshAfterAnnotationMutation(accepted.image.id);
    if (openedCandidate && accepted.annotation_id) {
      let activeAnnotationId = accepted.annotation_id;
      recordEditorAction({
        label: `accept review candidate #${candidateId}`,
        undo: async () => {
          await deleteAnnotationLocal(activeAnnotationId);
          const reopened = await api.reopenReviewCandidate(candidateId);
          setReviewCandidates((items) => items.some((item) => item.id === reopened.id) ? items : [reopened, ...items]);
          setCandidates((items) => items.some((item) => item.reviewCandidateId === candidateId) ? items : [openedCandidate, ...items]);
          setSelectedCandidate(openedCandidate.localId);
          await refreshAfterAnnotationMutation(accepted.image.id, accepted.image.id);
        },
        redo: async () => {
          const redone = await api.acceptReviewCandidate(candidateId);
          activeAnnotationId = redone.annotation_id ?? activeAnnotationId;
          setReviewCandidates((items) => items.filter((item) => item.id !== candidateId));
          setCandidates((items) => items.filter((item) => item.reviewCandidateId !== candidateId));
          setSelectedCandidate(null);
          await refreshAfterAnnotationMutation(accepted.image.id);
        }
      });
    }
    setStatus(`Accepted review candidate #${candidateId}.`);
  }

  async function rejectReviewCandidate(candidateId: number) {
    const openedCandidate = candidates.find((item) => item.reviewCandidateId === candidateId);
    await api.rejectReviewCandidate(candidateId);
    setReviewCandidates((items) => items.filter((item) => item.id !== candidateId));
    setCandidates((items) => items.filter((item) => item.reviewCandidateId !== candidateId));
    if (selectedCandidateObj?.reviewCandidateId === candidateId) {
      setSelectedCandidate(null);
      setEditMaskData(null);
    }
    if (openedCandidate) {
      recordEditorAction({
        label: `reject review candidate #${candidateId}`,
        undo: async () => {
          const reopened = await api.reopenReviewCandidate(candidateId);
          setReviewCandidates((items) => items.some((item) => item.id === reopened.id) ? items : [reopened, ...items]);
          setCandidates((items) => items.some((item) => item.reviewCandidateId === candidateId) ? items : [openedCandidate, ...items]);
          setSelectedCandidate(openedCandidate.localId);
        },
        redo: async () => {
          await api.rejectReviewCandidate(candidateId);
          setReviewCandidates((items) => items.filter((item) => item.id !== candidateId));
          setCandidates((items) => items.filter((item) => item.reviewCandidateId !== candidateId));
          setSelectedCandidate(null);
        }
      });
    }
    setStatus(`Rejected review candidate #${candidateId}.`);
  }

  async function openReviewCandidate(candidateId: number) {
    const opened = await api.openReviewCandidate(candidateId);
    navigateToImage(opened.candidate.image);
    if (currentImage?.id === opened.candidate.image.id) {
      addReviewCandidateToEditor(opened);
    } else {
      setPendingReviewOpen(opened);
    }
    setRightTab("editor");
    setStatus(`Loaded review candidate #${candidateId} into the editor.`);
  }

  async function loadReviewCandidatesForImage(imageId: number, imageReviewCandidates: ReviewCandidateRecord[]) {
    const existingIds = new Set(
      candidates
        .filter((candidate) => candidate.imageId === imageId && candidate.reviewCandidateId)
        .map((candidate) => candidate.reviewCandidateId as number)
    );
    const toLoad = imageReviewCandidates.filter(
      (candidate) => !existingIds.has(candidate.id) && !loadingReviewCandidateIdsRef.current.has(candidate.id)
    );
    if (toLoad.length === 0) return;

    toLoad.forEach((candidate) => loadingReviewCandidateIdsRef.current.add(candidate.id));
    try {
      const openedItems = (
        await Promise.all(
          toLoad.map(async (candidate) => {
            try {
              return await api.openReviewCandidate(candidate.id);
            } catch {
              return null;
            }
          })
        )
      ).filter((item): item is { candidate: ReviewCandidateRecord; mask_png: string } => Boolean(item));
      if (activeImageIdRef.current !== imageId || openedItems.length === 0) return;

      setCandidates((items) => {
        const openReviewIds = new Set(
          items
            .filter((candidate) => candidate.imageId === imageId && candidate.reviewCandidateId)
            .map((candidate) => candidate.reviewCandidateId as number)
        );
        const additions = openedItems
          .filter((opened) => opened.candidate.image.id === imageId && !openReviewIds.has(opened.candidate.id))
          .map((opened, index) => reviewCandidateToEditorCandidate(opened, Date.now() + index));
        return additions.length > 0 ? [...items, ...additions] : items;
      });
    } finally {
      toLoad.forEach((candidate) => loadingReviewCandidateIdsRef.current.delete(candidate.id));
    }
  }

  function reviewCandidateToEditorCandidate(opened: { candidate: ReviewCandidateRecord; mask_png: string }, localId = Date.now()): Candidate {
    return {
      localId,
      imageId: opened.candidate.image.id,
      reviewCandidateId: opened.candidate.id,
      name: opened.candidate.category_name,
      visible: true,
      mask_png: opened.mask_png,
      bbox: opened.candidate.bbox,
      area: opened.candidate.area,
      score: opened.candidate.score,
      prompt_type: "bulk_review",
      annotation: null
    };
  }

  function addReviewCandidateToEditor(opened: { candidate: ReviewCandidateRecord; mask_png: string }) {
    const existing = candidates.find((item) => item.reviewCandidateId === opened.candidate.id);
    if (existing) {
      setSelectedCandidate(existing.localId);
      setSelectedAnnotation(null);
      setStatus(`Review candidate #${opened.candidate.id} is already open in the editor.`);
      return;
    }
    const candidate = reviewCandidateToEditorCandidate(opened);
    setCandidates((items) => [candidate, ...items.filter((item) => item.imageId !== candidate.imageId || item.localId !== candidate.localId)]);
    setSelectedCandidate(candidate.localId);
    setSelectedAnnotation(null);
    setTool("view");
  }

  async function loadExportWorkspace() {
    if (exportLoading) return;
    setExportLoading(true);
    try {
      const response = await api.exportWorkspaceScan(exportRoot.trim());
      const nextFolders = response.folders.map((folder) => ({
        folder: folder.folder,
        images: folder.images,
        annotatedImages: folder.images.map((image) => ({
          id: image.id,
          file_name: image.file_name,
          width: image.width,
          height: image.height,
          accepted_object_count: image.accepted_object_count ?? image.annotation_count ?? 0,
          matching_class_count: image.matching_class_count ?? null
        })),
        annotationCount: folder.annotation_count,
      }));

      setExportFolders(nextFolders);
      setExportSplitPlans((current) => {
        const next: Record<string, ExportSplitPlan> = {};
        for (const folder of nextFolders) {
          const existing = current[folder.folder];
          if (existing) next[folder.folder] = existing;
        }
        return next;
      });
      setExportCombineSelection((current) => current.filter((folder) => nextFolders.some((item) => item.folder === folder)));
      setStatus(`Loaded ${nextFolders.length} annotated folders from ${response.root}.`);
    } finally {
      setExportLoading(false);
    }
  }

  function splitFolderImages(images: ImageRecord[]): ExportSplitPlan {
    const sorted = [...images].sort((a, b) => a.file_name.localeCompare(b.file_name));
    const total = sorted.length;
    if (total === 0) {
      return { train: [], val: [], test: [] };
    }
    const trainCount = Math.max(0, Math.min(total, Math.round((total * exportTrainPercent) / 100)));
    const valCount = Math.max(0, Math.min(total - trainCount, Math.round((total * exportValPercent) / 100)));
    const testCount = Math.max(0, total - trainCount - valCount);
    const train = sorted.slice(0, trainCount);
    const val = sorted.slice(trainCount, trainCount + valCount);
    const test = sorted.slice(trainCount + valCount, trainCount + valCount + testCount);
    return { train, val, test };
  }

  function splitFolder(folder: string) {
    const folderRecord = exportFolderMap.get(folder);
    if (!folderRecord) return;
    setExportSplitPlans((current) => ({
      ...current,
      [folder]: splitFolderImages(folderRecord.images)
    }));
    setStatus(`Split ${folder} into train/val/test.`);
  }

  function autoSplitFolders() {
    if (exportFolders.length === 0) {
      setStatus("Load annotated folders first.");
      return;
    }
    const next: Record<string, ExportSplitPlan> = {};
    for (const folder of exportFolders) {
      next[folder.folder] = splitFolderImages(folder.images);
    }
    setExportSplitPlans(next);
    setStatus(`Split ${exportFolders.length} folders using ${exportTrainPercent}/${exportValPercent}/${exportTestPercent} preferences.`);
  }

  function toggleExportCombineSelection(folder: string) {
    setExportCombineSelection((current) =>
      current.includes(folder) ? current.filter((item) => item !== folder) : [...current, folder]
    );
  }

  function combineSelectedFolders() {
    if (exportCombineSelection.length === 0) {
      setStatus("Select one or more folders to combine.");
      return;
    }
    const nextGroup = {
      id: Date.now(),
      name: `Combined dataset ${exportCombinedGroups.length + 1}`,
      folders: [...exportCombineSelection].sort((a, b) => a.localeCompare(b))
    };
    setExportCombinedGroups((groups) => [...groups, nextGroup]);
    setExportCombineSelection([]);
    setStatus(`Created ${nextGroup.name} from ${nextGroup.folders.length} folder(s).`);
  }

  async function runExport() {
    const assignedCount = Object.values(exportSplitPlans).reduce(
      (total, plan) => total + plan.train.length + plan.val.length + plan.test.length,
      0
    );
    if (exportFolders.length > 0 && assignedCount === 0) {
      setStatus("Split folders before exporting so COCO can follow train/val/test assignments.");
      return;
    }
    const payload: ExportCocoRequest = exportFolders.length > 0
      ? {
          root: exportRoot.trim(),
          folder_splits: exportFolders
            .filter((folder) => exportSplitPlans[folder.folder])
            .map((folder) => {
              const plan = exportSplitPlans[folder.folder];
              return {
                folder: folder.folder,
                splits: {
                  train: plan.train.map((image) => image.id),
                  val: plan.val.map((image) => image.id),
                  test: plan.test.map((image) => image.id),
                }
              };
            })
        }
      : {
          splits: {
            train: Object.values(exportSplitPlans).flatMap((plan) => plan.train.map((image) => image.id)),
            val: Object.values(exportSplitPlans).flatMap((plan) => plan.val.map((image) => image.id)),
            test: Object.values(exportSplitPlans).flatMap((plan) => plan.test.map((image) => image.id)),
          }
        };
    setStatus("Validating and exporting...");
    const result = await api.exportCoco(assignedCount > 0 ? payload : undefined);
    const splitFiles = Object.keys(result.split_coco_jsons ?? {}).length;
    const folderCount = result.folder_exports?.length ?? 0;
    setStatus(`Exported ${result.mask_count} masks into ${splitFiles || 1} COCO file(s)${folderCount ? ` across ${folderCount} folder(s)` : ""} at ${result.export_dir}`);
  }

  async function runQa() {
    const issues = await api.validate();
    setStatus(issues.length ? `${issues.length} QA issues. First: ${issues[0].message}` : "QA passed.");
  }

  function onPointerDown(evt: React.MouseEvent) {
    if (evt.button === 1) {
      evt.preventDefault();
      setPanning(true);
      panStartRef.current = { x: evt.clientX, y: evt.clientY };
      return;
    }
    if (evt.button !== 0) return;
    if (tool === "view") {
      clearSelection();
      setPanning(true);
      panStartRef.current = { x: evt.clientX, y: evt.clientY };
      return;
    }
    const point = canvasPoint(evt);
    if (!point) {
      clearSelection();
      return;
    }
    const existingPromptIndex = findNearestPoint(promptPoints, point);
    if (existingPromptIndex !== null && (tool === "point_pos" || tool === "point_neg")) {
      pushCandidateHistory();
      setDragPromptIndex(existingPromptIndex);
      return;
    }
    const existingPolygonIndex = findNearestPoint(polygonPoints, point);
    if (existingPolygonIndex !== null && tool === "polygon") {
      pushCandidateHistory();
      setDragPolygonIndex(existingPolygonIndex);
      return;
    }
    if (tool === "point_pos" || tool === "point_neg") {
      void runPoint(point, tool === "point_pos" ? 1 : 0);
      return;
    }
    if (tool === "polygon") {
      pushCandidateHistory();
      setPolygonPoints((points) => [...points, point]);
      setStatus("Polygon point added. Finish Polygon saves it as a candidate.");
      return;
    }
    if ((tool === "brush" || tool === "erase") && (selectedCandidateObj || selectedAnnotation)) {
      if (selectedCandidateObj) pushCandidateHistory();
      setPainting(true);
      setEditingTarget(selectedCandidateObj ? "candidate" : "annotation");
      paintMask(evt);
      return;
    }
    if ((tool === "brush" || tool === "erase") && !selectedCandidateObj && !selectedAnnotation) {
      setStatus("Select a candidate or saved object before brushing.");
      return;
    }
    if (tool !== "box") return;
    setBoxStart(point);
  }

  function onPointerMove(evt: React.MouseEvent) {
    if (panning && panStartRef.current) {
      const dx = evt.clientX - panStartRef.current.x;
      const dy = evt.clientY - panStartRef.current.y;
      panStartRef.current = { x: evt.clientX, y: evt.clientY };
      setPan((previous) => ({ x: previous.x + dx, y: previous.y + dy }));
      return;
    }
    const point = canvasPoint(evt);
    setBrushCursor((tool === "brush" || tool === "erase") && point ? point : null);
    if (point && resizeDrag) {
      void updateResizeDrag(point);
      return;
    }
    if (point && dragPromptIndex !== null) {
      setPromptPoints((points) => points.map((item, index) => (index === dragPromptIndex ? { ...item, ...point } : item)));
      return;
    }
    if (point && dragPolygonIndex !== null) {
      setPolygonPoints((points) => points.map((item, index) => (index === dragPolygonIndex ? point : item)));
      return;
    }
    if (painting && (tool === "brush" || tool === "erase")) {
      paintMask(evt);
      return;
    }
    if (tool !== "box" || !boxStart) return;
    if (!point) return;
    setDraftBox({
      x: Math.min(boxStart.x, point.x),
      y: Math.min(boxStart.y, point.y),
      width: Math.abs(point.x - boxStart.x),
      height: Math.abs(point.y - boxStart.y)
    });
  }

  async function onPointerUp() {
    if (panning) {
      setPanning(false);
      panStartRef.current = null;
      return;
    }
    if (resizeDrag) {
      const target = resizeDrag.target;
      setResizeDrag(null);
      if (target === "annotation" && selectedAnnotation && editMaskDataRef.current && currentImage) {
        await api.replaceMask(selectedAnnotation, editMaskDataRef.current);
        await refreshAnnotations(currentImage.id);
        setStatus("Saved adjusted object outline.");
      } else {
        setStatus("Adjusted selected candidate outline.");
      }
      setEditingTarget(null);
      return;
    }
    if (dragPromptIndex !== null) {
      const nextPoints = [...promptPointsRef.current];
      setDragPromptIndex(null);
      await runPointSet(nextPoints);
      return;
    }
    if (dragPolygonIndex !== null) {
      setDragPolygonIndex(null);
      setStatus("Polygon point moved.");
      return;
    }
    if (painting) {
      setPainting(false);
      await saveEditedMask();
      setEditingTarget(null);
      return;
    }
    if (tool === "box" && draftBox && draftBox.width > 4 && draftBox.height > 4) {
      await runBoxPrompt(draftBox);
    }
    setBoxStart(null);
    setDraftBox(null);
  }

  function onPointerLeave() {
    setBrushCursor(null);
    if (!panning) return;
    setPanning(false);
    panStartRef.current = null;
  }

  function finishPolygon() {
    if (!currentImage || polygonPoints.length < 3) {
      setStatus("Polygon needs at least three points.");
      return;
    }
    const canvas = document.createElement("canvas");
    canvas.width = currentImage.width;
    canvas.height = currentImage.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "white";
    ctx.beginPath();
    polygonPoints.forEach((point, index) => {
      if (index === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    });
    ctx.closePath();
    ctx.fill();
    const xs = polygonPoints.map((point) => point.x);
    const ys = polygonPoints.map((point) => point.y);
    const x0 = Math.min(...xs);
    const y0 = Math.min(...ys);
    const x1 = Math.max(...xs);
    const y1 = Math.max(...ys);
    const manualCandidate: Candidate = {
      localId: Date.now(),
      imageId: currentImage.id,
      name: activeClassName,
      visible: true,
      mask_png: canvas.toDataURL("image/png"),
      bbox: [x0, y0, x1 - x0, y1 - y0],
      area: Math.round(Math.abs(polygonPoints.reduce((sum, point, index) => {
        const next = polygonPoints[(index + 1) % polygonPoints.length];
        return sum + point.x * next.y - next.x * point.y;
      }, 0)) / 2),
      score: null,
      prompt_type: "polygon",
      annotation: null
    };
    pushCandidateHistory();
    setCandidates((items) => [...items, manualCandidate]);
    setSelectedCandidate(manualCandidate.localId);
    setPolygonPoints([]);
    setStatus("Manual polygon added as a candidate. Accept to save it.");
  }

  async function drawOverlay() {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    if (!canvas || !img || !currentImage) return;
    const seq = ++renderSeqRef.current;
    canvas.width = currentImage.width;
    canvas.height = currentImage.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const visibleAnnotations = annotations.filter((item) => item.visible && item.image_id === currentImage.id);
    const visibleCandidates = candidates.filter((item) => item.visible && item.imageId === currentImage.id);

    for (const annotation of visibleAnnotations) {
      if (seq !== renderSeqRef.current) return;
      const mask = annotationMasks[annotation.id];
      if (!mask) continue;
      await drawMask(ctx, mask, hexToRgba(COLORS[annotation.id % COLORS.length], annotation.id === selectedAnnotation ? 0.28 : 0.16), seq);
      await drawMaskOutline(ctx, mask, COLORS[annotation.id % COLORS.length], annotation.id === selectedAnnotation ? 3 : 2, seq);
    }
    for (const candidate of visibleCandidates) {
      if (seq !== renderSeqRef.current) return;
      const index = candidates.findIndex((item) => item.localId === candidate.localId);
      await drawMask(ctx, candidate.mask_png, hexToRgba(COLORS[index % COLORS.length], candidate.localId === selectedCandidate ? 0.32 : 0.14), seq);
      await drawMaskOutline(ctx, candidate.mask_png, COLORS[index % COLORS.length], candidate.localId === selectedCandidate ? 3 : 2, seq);
    }
    if (editMaskData && selectedAnnotation && !selectedCandidateObj && selectedAnnotationObj?.visible) {
      await drawMask(ctx, editMaskData, "rgba(56, 116, 214, 0.42)", seq);
      await drawMaskOutline(ctx, editMaskData, "#3874d6", 3, seq);
    }
    if (showCanvasLabels) {
      for (const annotation of visibleAnnotations) {
        drawAnnotationLabel(ctx, annotation);
      }
      for (const candidate of visibleCandidates) {
        const index = candidates.findIndex((item) => item.localId === candidate.localId);
        drawCandidateLabel(ctx, candidate, index);
      }
    }
    drawPromptPoints(ctx);
    drawPolygon(ctx);
  }

  function ensureMaskCanvas() {
    if (!currentImage) return null;
    if (!maskCanvasRef.current) maskCanvasRef.current = document.createElement("canvas");
    const maskCanvas = maskCanvasRef.current;
    if (maskCanvas.width !== currentImage.width || maskCanvas.height !== currentImage.height) {
      maskCanvas.width = currentImage.width;
      maskCanvas.height = currentImage.height;
      const ctx = maskCanvas.getContext("2d");
      ctx?.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    }
    return maskCanvas;
  }

  function hydrateMaskCanvas(dataUrl: string) {
    const imageId = currentImage?.id ?? null;
    const maskCanvas = ensureMaskCanvas();
    if (!maskCanvas) return;
    const ctx = maskCanvas.getContext("2d");
    if (!ctx) return;
    const img = new Image();
    img.onload = () => {
      if (activeImageIdRef.current !== imageId) return;
      ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
      ctx.drawImage(img, 0, 0, maskCanvas.width, maskCanvas.height);
      drawOverlay();
    };
    img.src = dataUrl;
  }

  function paintMask(evt: React.MouseEvent) {
    const point = canvasPoint(evt);
    const maskCanvas = ensureMaskCanvas();
    if (!point || !maskCanvas) return;
    const ctx = maskCanvas.getContext("2d");
    if (!ctx) return;
    ctx.save();
    ctx.globalCompositeOperation = "source-over";
    ctx.fillStyle = tool === "brush" ? "white" : "black";
    ctx.beginPath();
    ctx.arc(point.x, point.y, brushSize, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    setEditMaskData(maskCanvas.toDataURL("image/png"));
  }

  async function saveEditedMask() {
    if (!currentImage) return;
    const maskCanvas = ensureMaskCanvas();
    if (!maskCanvas) return;
    const maskPng = maskCanvas.toDataURL("image/png");
    if (editingTarget === "candidate" && selectedCandidateObj) {
      setCandidates((items) =>
        items.map((candidate) =>
          candidate.localId === selectedCandidateObj.localId ? { ...candidate, mask_png: maskPng } : candidate
        )
      );
      setStatus("Candidate mask edited. Accept to save it as an object.");
      return;
    }
    if (selectedAnnotation) {
      await api.replaceMask(selectedAnnotation, maskPng);
      await refreshAnnotations(currentImage.id);
      setStatus("Saved object mask edit.");
    }
  }

  function drawMask(ctx: CanvasRenderingContext2D, dataUrl: string, fill: string, seq: number) {
    return new Promise<void>((resolve) => {
      const mask = new Image();
      mask.onload = () => {
        if (seq !== renderSeqRef.current) {
          resolve();
          return;
        }
        const pixels = maskPixels(mask, ctx.canvas.width, ctx.canvas.height);
        if (!pixels) {
          resolve();
          return;
        }
        const overlay = ctx.createImageData(pixels.width, pixels.height);
        const parts = fill.match(/[\d.]+/g)?.map(Number) ?? [28, 161, 122, 0.45];
        const [r, g, b, alpha = 0.45] = parts;
        for (let i = 0; i < pixels.data.length; i += 4) {
          if (isMaskPixel(pixels.data, i)) {
            overlay.data[i] = r;
            overlay.data[i + 1] = g;
            overlay.data[i + 2] = b;
            overlay.data[i + 3] = Math.round(alpha * 255);
          }
        }
        const overlayCanvas = document.createElement("canvas");
        overlayCanvas.width = pixels.width;
        overlayCanvas.height = pixels.height;
        const overlayCtx = overlayCanvas.getContext("2d");
        if (!overlayCtx) {
          resolve();
          return;
        }
        overlayCtx.putImageData(overlay, 0, 0);
        if (seq === renderSeqRef.current) ctx.drawImage(overlayCanvas, 0, 0);
        resolve();
      };
      mask.onerror = () => resolve();
      mask.src = dataUrl;
    });
  }

  function drawMaskOutline(ctx: CanvasRenderingContext2D, dataUrl: string, color: string, lineWidth: number, seq: number) {
    return new Promise<void>((resolve) => {
      const mask = new Image();
      mask.onload = () => {
        if (seq !== renderSeqRef.current) {
          resolve();
          return;
        }
        const pixels = maskPixels(mask, ctx.canvas.width, ctx.canvas.height);
        if (!pixels) {
          resolve();
          return;
        }
        const outline = ctx.createImageData(pixels.width, pixels.height);
        for (let y = 0; y < pixels.height; y += 1) {
          for (let x = 0; x < pixels.width; x += 1) {
            const index = (y * pixels.width + x) * 4;
            if (!isMaskPixel(pixels.data, index)) continue;
            if (!isMaskPixelAt(pixels, x - 1, y) || !isMaskPixelAt(pixels, x + 1, y) || !isMaskPixelAt(pixels, x, y - 1) || !isMaskPixelAt(pixels, x, y + 1)) {
              outline.data[index] = 255;
              outline.data[index + 1] = 255;
              outline.data[index + 2] = 255;
              outline.data[index + 3] = 255;
            }
          }
        }
        const outlineCanvas = document.createElement("canvas");
        outlineCanvas.width = pixels.width;
        outlineCanvas.height = pixels.height;
        const outlineCtx = outlineCanvas.getContext("2d");
        if (!outlineCtx) {
          resolve();
          return;
        }
        outlineCtx.putImageData(outline, 0, 0);
        outlineCtx.globalCompositeOperation = "source-in";
        outlineCtx.fillStyle = color;
        outlineCtx.fillRect(0, 0, pixels.width, pixels.height);
        outlineCtx.globalCompositeOperation = "source-over";
        if (seq !== renderSeqRef.current) {
          resolve();
          return;
        }
        ctx.save();
        ctx.globalCompositeOperation = "source-over";
        ctx.shadowColor = "rgba(0,0,0,0.65)";
        ctx.shadowBlur = 2;
        ctx.drawImage(outlineCanvas, 0, 0);
        ctx.restore();
        if (lineWidth > 1) {
          ctx.save();
          ctx.globalAlpha = 0.9;
          ctx.filter = `drop-shadow(0 0 ${lineWidth}px ${color})`;
          if (seq === renderSeqRef.current) ctx.drawImage(outlineCanvas, 0, 0);
          ctx.restore();
        }
        resolve();
      };
      mask.onerror = () => resolve();
      mask.src = dataUrl;
    });
  }

  function drawAnnotationLabel(ctx: CanvasRenderingContext2D, annotation: AnnotationRecord) {
    const [x, y] = annotation.bbox;
    const color = COLORS[annotation.id % COLORS.length];
    const label = `obj=${annotation.id}, ${annotation.category_name ?? "food"}`;
    ctx.save();
    ctx.font = "13px system-ui";
    const textWidth = ctx.measureText(label).width + 8;
    ctx.fillStyle = "rgba(255,255,255,0.92)";
    ctx.fillRect(x, Math.max(0, y - 18), textWidth, 18);
    ctx.fillStyle = color;
    ctx.fillText(label, x + 4, Math.max(13, y - 5));
    ctx.restore();
  }

  function drawCandidateLabel(ctx: CanvasRenderingContext2D, candidate: Candidate, index: number) {
    const [x, y] = candidate.bbox;
    const color = COLORS[index % COLORS.length];
    const label = `id=${index + 1}, prob=${candidate.score == null ? "manual" : candidate.score.toFixed(2)}`;
    ctx.save();
    ctx.font = "13px system-ui";
    const textWidth = ctx.measureText(label).width + 8;
    ctx.fillStyle = "rgba(255,255,255,0.92)";
    ctx.fillRect(x, Math.max(0, y - 18), textWidth, 18);
    ctx.fillStyle = color;
    ctx.fillText(label, x + 4, Math.max(13, y - 5));
    ctx.restore();
  }

  function drawResizeHandles(ctx: CanvasRenderingContext2D, bbox: number[], color: string) {
    ctx.save();
    ctx.fillStyle = "#ffffff";
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    for (const handle of bboxHandles(bbox)) {
      ctx.beginPath();
      ctx.rect(handle.x - 5, handle.y - 5, 10, 10);
      ctx.fill();
      ctx.stroke();
    }
    ctx.restore();
  }

  function findNearestResizeHandle(point: Point): ResizeDrag | null {
    const target =
      selectedCandidateObj && selectedCandidateObj.visible
        ? { kind: "candidate" as const, id: selectedCandidateObj.localId, bbox: selectedCandidateObj.bbox, mask: selectedCandidateObj.mask_png }
        : selectedAnnotationObj?.visible && editMaskData
          ? { kind: "annotation" as const, id: selectedAnnotationObj.id, bbox: selectedAnnotationObj.bbox, mask: editMaskData }
          : null;
    if (!target) return null;
    for (const handle of bboxHandles(target.bbox)) {
      const dx = handle.x - point.x;
      const dy = handle.y - point.y;
      if (dx * dx + dy * dy <= 14 * 14) {
        return {
          target: target.kind,
          id: target.id,
          handle: handle.name,
          originalBbox: target.bbox,
          originalMask: target.mask
        };
      }
    }
    return null;
  }

  async function updateResizeDrag(point: Point) {
    if (!currentImage || !resizeDrag) return;
    const bbox = resizedBbox(resizeDrag.originalBbox, resizeDrag.handle, point, currentImage.width, currentImage.height);
    const maskPng = await transformMaskToBbox(resizeDrag.originalMask, resizeDrag.originalBbox, bbox, currentImage.width, currentImage.height);
    if (resizeDrag.target === "candidate") {
      setCandidates((items) =>
        items.map((candidate) =>
          candidate.localId === resizeDrag.id ? { ...candidate, bbox, mask_png: maskPng } : candidate
        )
      );
      return;
    }
    setEditMaskData(maskPng);
    setAnnotationMasks((previous) => ({ ...previous, [resizeDrag.id]: maskPng }));
    setAnnotations((items) => items.map((annotation) => (annotation.id === resizeDrag.id ? { ...annotation, bbox } : annotation)));
  }

  function drawPromptPoints(ctx: CanvasRenderingContext2D) {
    promptPoints.forEach((point) => {
      ctx.save();
      ctx.fillStyle = point.label ? "#23c37a" : "#e23b3b";
      ctx.strokeStyle = "white";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(point.x, point.y, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    });
  }

  function drawPolygon(ctx: CanvasRenderingContext2D) {
    if (polygonPoints.length === 0) return;
    ctx.save();
    ctx.strokeStyle = "#24d39a";
    ctx.fillStyle = "rgba(36, 211, 154, 0.16)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    polygonPoints.forEach((point, index) => {
      if (index === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    });
    if (polygonPoints.length > 2) ctx.closePath();
    ctx.stroke();
    if (polygonPoints.length > 2) ctx.fill();
    polygonPoints.forEach((point) => {
      ctx.beginPath();
      ctx.arc(point.x, point.y, 5, 0, Math.PI * 2);
      ctx.fillStyle = "#ffffff";
      ctx.fill();
      ctx.strokeStyle = "#24d39a";
      ctx.stroke();
    });
    ctx.restore();
  }

  function candidateIndex(candidate: Candidate) {
    return candidates.findIndex((item) => item.localId === candidate.localId) + 1;
  }

  const pendingCount = useMemo(() => annotations.filter((annotation) => annotation.status === "pending").length, [annotations]);
  const combinableObjectCount = useMemo(
    () => annotations.filter((annotation) => annotation.image_id === currentImage?.id && annotation.status === "accepted").length,
    [annotations, currentImage?.id]
  );
  const selectedJob = bulkJobs.find((job) => job.id === selectedJobId) ?? null;
  const reviewGroups = useMemo(() => {
    const groups = new Map<number, { image: ImageRecord; candidates: ReviewCandidateRecord[] }>();
    for (const candidate of reviewCandidates) {
      const group = groups.get(candidate.image.id) ?? { image: candidate.image, candidates: [] };
      group.candidates.push(candidate);
      groups.set(candidate.image.id, group);
    }
    return Array.from(groups.values());
  }, [reviewCandidates]);

  return (
    <main className="app" onMouseDownCapture={unselectOutsideImage}>
      <aside className="left">
        <section className="panel project">
          <h1>FoodSeg</h1>
          <div className="path-row">
            <input value={projectPath} onChange={(event) => setProjectPath(event.target.value)} />
            <button onClick={openProject}>Open</button>
          </div>
          <p>{project || "No project open"}</p>
        </section>

        <section className="panel image-list">
          <div className="section-title">Images</div>
          <label className="field-label">
            Class Name
            <input
              className="image-search"
              value={annotationClassName}
              onChange={(event) => setAnnotationClassName(event.target.value)}
              placeholder="Class name for new masks"
            />
          </label>
          <button className="panel-action" onClick={() => setImageFiltersOpen((open) => !open)}>
            {imageFiltersOpen ? "Hide Image Filters" : "Show Image Filters"}{hasImageFilters ? " *" : ""}
          </button>
          {imageFiltersOpen && (
            <div className="image-filters">
              <input
                value={imageFilenameFilter}
                onChange={(event) => setImageFilenameFilter(event.target.value)}
                placeholder="Search filenames"
              />
              <label>
                Mask Status
                <select value={imageMaskFilter} onChange={(event) => setImageMaskFilter(event.target.value as ImageMaskFilter)}>
                  <option value="all">All</option>
                  <option value="with_masks">With masks</option>
                  <option value="without_masks">Without masks</option>
                </select>
              </label>
              <label>
                Class Filter
                <div className="input-action-row">
                  <input
                    value={imageClassFilter}
                    onChange={(event) => setImageClassFilter(event.target.value)}
                    placeholder="Exact class name"
                  />
                  <button onClick={() => setImageClassFilter(activeClassName)}>Use Class</button>
                </div>
              </label>
              <label>
                Object Count
                <div className="count-filter-row">
                  <select value={imageCountOp} onChange={(event) => setImageCountOp(event.target.value as ImageCountOperator)}>
                    <option value="lt">&lt;</option>
                    <option value="lte">&lt;=</option>
                    <option value="eq">=</option>
                    <option value="gte">&gt;=</option>
                    <option value="gt">&gt;</option>
                  </select>
                  <input
                    type="number"
                    min="0"
                    value={imageCountValue}
                    onChange={(event) => setImageCountValue(event.target.value)}
                    placeholder="Any"
                  />
                </div>
              </label>
              <button onClick={clearImageFilters} disabled={!hasImageFilters}>Clear Filters</button>
            </div>
          )}
          <p className="hint">
            {indexStatus?.status === "indexing"
              ? `Indexing... ${indexStatus.indexed_count} indexed`
              : `${images.length}/${imageTotal} loaded`}
          </p>
          <div className="image-scroll" onScroll={onImageListScroll}>
            {images.map((image, index) => (
              <button
                key={image.id}
                className={index === currentIndex ? "image-item active" : "image-item"}
                onClick={() => setCurrentIndex(index)}
              >
                <span>{image.file_name}</span>
                <small>
                  {image.width}x{image.height} | {image.accepted_object_count} obj
                  {imageClassFilter.trim() && image.matching_class_count != null ? ` | ${image.matching_class_count} class` : ""}
                </small>
              </button>
            ))}
            {imageLoading && <p className="hint">Loading images...</p>}
            {imageHasMore && !imageLoading && <button onClick={() => loadImagePage(images.length, false)}>Load More</button>}
          </div>
        </section>
      </aside>

      <section className="center">
        <div className="topbar">
          <button onClick={goPreviousImage} title="Previous image">
            <ChevronLeft size={18} />
          </button>
          <div className="filename">{currentImage?.file_name ?? "No image"}</div>
          <button onClick={goNextImage} title="Next image">
            <ChevronRight size={18} />
          </button>
          <div className="toolbar">
            {TOOL_ITEMS.map((item) => (
              <button
                key={item.id}
                className={tool === item.id ? "tool active" : "tool"}
                onClick={() => selectTool(item.id)}
                title={item.label}
              >
                {item.icon}
              </button>
            ))}
          </div>
          <button onClick={undoUniversal} title="Undo" disabled={!canUndo}>
            <Undo2 size={18} />
          </button>
          <button onClick={redoUniversal} title="Redo" disabled={!canRedo}>
            <Redo2 size={18} />
          </button>
          <button onClick={() => zoomAt(zoom / 1.2)} title="Zoom out">
            <ZoomOut size={18} />
          </button>
          <button onClick={() => resetViewport()} title="Fit image">
            <Maximize2 size={18} />
          </button>
          <button onClick={() => zoomAt(zoom * 1.2)} title="Zoom in">
            <ZoomIn size={18} />
          </button>
          <input
            className="prompt"
            value={promptText}
            onChange={(event) => setPromptText(event.target.value)}
            placeholder="SAM text prompt"
            title="SAM text prompt. New masks are named with the Class Name field on the left."
          />
          <button onClick={runTextPrompt} title="Run text prompt">
            <Wand2 size={18} />
          </button>
          {tool === "polygon" && <button onClick={finishPolygon}>Finish Polygon</button>}
          {selectedCandidateObj && (
            <button onClick={acceptSelectedCandidate} title="Save selected candidate as object">
              <Check size={16} />
              Accept Selected
            </button>
          )}
          {candidates.length > 1 && <button onClick={acceptAllCandidates}>Accept All</button>}
          {candidates.length > 0 && <button onClick={() => clearCandidates()}>Clear Candidates</button>}
        </div>

        <div
          ref={stageRef}
          className={tool === "view" ? "stage view-mode" : "stage"}
          onMouseDown={onPointerDown}
          onMouseMove={onPointerMove}
          onMouseUp={onPointerUp}
          onMouseLeave={onPointerLeave}
          onAuxClick={(event) => {
            if (event.button === 1) event.preventDefault();
          }}
          onWheel={onWheelStage}
        >
          {currentImage && (
            <div
              className="image-frame"
              style={{
                width: `${currentImage.width}px`,
                height: `${currentImage.height}px`,
                transform: `translate(${pan.x}px, ${pan.y}px) scale(${viewportScale})`
              }}
            >
              <img
                ref={imageRef}
                src={imageUrl(currentImage.id, projectKey)}
                onLoad={() => {
                  resetViewport();
                  drawOverlay();
                }}
                draggable={false}
                onDragStart={(event) => event.preventDefault()}
              />
              <canvas ref={canvasRef} />
              {(tool === "brush" || tool === "erase") && brushCursor && (
                <div
                  className={tool === "erase" ? "brush-cursor erase" : "brush-cursor"}
                  style={{
                    left: `${brushCursor.x - brushSize}px`,
                    top: `${brushCursor.y - brushSize}px`,
                    width: `${brushSize * 2}px`,
                    height: `${brushSize * 2}px`
                  }}
                />
              )}
              {draftBox && currentImage && (
                <div
                  className="draft-box"
                  style={{
                    left: `${(draftBox.x / currentImage.width) * 100}%`,
                    top: `${(draftBox.y / currentImage.height) * 100}%`,
                    width: `${(draftBox.width / currentImage.width) * 100}%`,
                    height: `${(draftBox.height / currentImage.height) * 100}%`
                  }}
                />
              )}
            </div>
          )}
          <div className="toast">{status}</div>
        </div>

        <div className="status">{status}</div>
      </section>

      <aside className="right">
        <div className="tabs">
          <button className={rightTab === "editor" ? "active" : ""} onClick={() => setRightTab("editor")}>Editor</button>
          <button className={rightTab === "bulk" ? "active" : ""} onClick={() => setRightTab("bulk")}>Bulk Review</button>
          <button className={rightTab === "export" ? "active" : ""} onClick={() => setRightTab("export")}>Export</button>
        </div>

        {rightTab === "editor" && (
          <>
            <section className="panel">
              <div className="section-title">View</div>
              <div className="compact-row">
                <button onClick={clearSelection} disabled={!selectedCandidate && !selectedAnnotation}>
                  <X size={16} />
                  Deselect
                </button>
                <button onClick={() => setShowCanvasLabels((value) => !value)}>
                  {showCanvasLabels ? <EyeOff size={16} /> : <Eye size={16} />}
                  {showCanvasLabels ? "Hide ID Labels" : "Show ID Labels"}
                </button>
              </div>
            </section>

            <section className="panel">
              <div className="section-title">Unsaved Candidates</div>
              <div className="prompt-filters">
                <div className="compact-row">
                  <select value={promptFilterMode} onChange={(event) => setPromptFilterMode(event.target.value as CandidateFilterMode)}>
                    <option value="top_k">Top K only</option>
                    <option value="threshold_top_k">Threshold + Top K</option>
                  </select>
                  <label>
                    Top K
                    <input type="number" min="1" max="20" value={promptTopK} onChange={(event) => setPromptTopK(Number(event.target.value))} />
                  </label>
                </div>
                {promptFilterMode === "threshold_top_k" && (
                  <label>
                    Confidence threshold
                    <input type="number" min="0" max="1" step="0.05" value={promptThreshold} onChange={(event) => setPromptThreshold(Number(event.target.value))} />
                  </label>
                )}
              </div>
              {candidates.length === 0 && <p className="hint">Run a prompt or draw a polygon. Accept saves a candidate as an object.</p>}
              {candidates.map((candidate, index) => (
                <div
                  key={candidate.localId}
                  className={candidate.localId === selectedCandidate ? "candidate active" : "candidate"}
                >
                  <button
                    className="candidate-main"
                    onClick={() => {
                      setSelectedCandidate(candidate.localId);
                      setSelectedAnnotation(null);
                    }}
                  >
                    <span style={{ borderColor: COLORS[index % COLORS.length] }}>id={index + 1}</span>
                    <input
                      className="inline-name"
                      value={candidate.name}
                      onChange={(event) => renameCandidate(candidate.localId, event.target.value)}
                      onClick={(event) => event.stopPropagation()}
                      title="Rename candidate"
                    />
                    {candidate.localId === selectedCandidate && <small className="selected-pill">Selected</small>}
                    <small>{candidate.score == null ? "manual" : candidate.score.toFixed(2)}</small>
                  </button>
                  <button title="Toggle candidate visibility" onClick={() => toggleCandidateVisibility(candidate.localId)}>
                    {candidate.visible ? <Eye size={16} /> : <EyeOff size={16} />}
                  </button>
                </div>
              ))}
            </section>

            <section className="panel">
              <div className="section-title">Saved Objects {pendingCount ? `(${pendingCount} pending)` : ""}</div>
              {annotations.length > 0 && (
                <button className="panel-action" onClick={showAllObjects}>
                  <Eye size={16} />
                  Show All Object Masks
                </button>
              )}
              {annotations.length > 0 && (
                <button
                  className="panel-action"
                  onClick={combineSavedObjectMasks}
                  disabled={combinableObjectCount < 2}
                  title="Union all accepted saved masks on this image into one object"
                >
                  <Plus size={16} />
                  Combine All Object Masks
                </button>
              )}
              {annotations.map((annotation) => (
                <div key={annotation.id} className={annotation.id === selectedAnnotation ? "object active" : "object"}>
                  <button
                    className="object-main"
                    onClick={() => {
                      setSelectedAnnotation(annotation.id);
                      setSelectedCandidate(null);
                    }}
                  >
                    <input
                      className="inline-name"
                      value={annotation.category_name ?? "food"}
                      onChange={(event) =>
                        setAnnotations((items) =>
                          items.map((item) => (item.id === annotation.id ? { ...item, category_name: event.target.value } : item))
                        )
                      }
                      onBlur={(event) => renameAnnotation(annotation, event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter") event.currentTarget.blur();
                      }}
                      onClick={(event) => event.stopPropagation()}
                      title={`Rename object #${annotation.id}`}
                    />
                    <small>{annotation.status} | {annotation.area}px | v{annotation.version}</small>
                    {annotation.id === selectedAnnotation && <small className="selected-pill">Selected</small>}
                  </button>
                  <button title="Toggle visibility" onClick={() => toggleAnnotationVisibility(annotation)}>
                    {annotation.visible ? <Eye size={16} /> : <EyeOff size={16} />}
                  </button>
                  {annotation.status === "pending" && <button onClick={() => acceptPending(annotation.id)}>Accept</button>}
                  {annotation.status === "pending" && <button onClick={() => rejectPending(annotation.id)}>Reject</button>}
                  <button title="Delete" onClick={() => removeAnnotation(annotation.id)}>
                    <Trash2 size={16} />
                  </button>
                </div>
              ))}
            </section>

            <section className="panel">
              <div className="section-title">Export / QA</div>
              <button onClick={runPropagate}>Propagate Selected</button>
              <button onClick={runQa}>Run QA</button>
              <button onClick={() => setRightTab("export")}>
                <Download size={16} />
                Export
              </button>
            </section>
          </>
        )}

        {rightTab === "bulk" && (
          <>
            <section className="panel">
              <div className="section-title">CLIP Index</div>
              <p className="hint">
                {clipStatus
                  ? `${clipStatus.available ? "Ready" : "Unavailable"} | ${clipStatus.indexed}/${clipStatus.total} indexed | ${clipStatus.model} on ${clipStatus.device}`
                  : "Open a project to check CLIP."}
              </p>
              {clipStatus?.error && <p className="hint error">{clipStatus.error}</p>}
              <div className="compact-row">
                <button onClick={() => runClipIndex(false)}>Index CLIP</button>
                <button onClick={() => runClipIndex(true)}>Rebuild</button>
              </div>
            </section>

            <section className="panel">
              <div className="section-title">CLIP Search</div>
              <div className="compact-row">
                <input value={searchText} onChange={(event) => setSearchText(event.target.value)} />
                <button onClick={runSearch} title="Text search">
                  <Search size={17} />
                </button>
              </div>
              <button onClick={runCurrentImageSearch}>Similar To Current Image</button>
              <div className="results">
                {searchResults.map((result) => (
                  <button key={result.image.id} onClick={() => navigateToImage(result.image)}>
                    {result.image.file_name}<small>{result.score.toFixed(3)}</small>
                  </button>
                ))}
              </div>
            </section>

            <section className="panel">
              <div className="section-title">Bulk Job</div>
              <div className="compact-row">
                <select value={bulkMode} onChange={(event) => setBulkMode(event.target.value as BulkMode)}>
                  <option value="all">All Images</option>
                  <option value="clip_filtered">CLIP Filtered</option>
                </select>
                <input value={bulkPrompt} onChange={(event) => setBulkPrompt(event.target.value)} />
              </div>
              <div className="compact-row">
                <label>
                  Threshold
                  <input type="number" min="0" max="1" step="0.05" value={bulkThreshold} onChange={(event) => setBulkThreshold(Number(event.target.value))} />
                </label>
                <label>
                  Top K
                    <input type="number" min="1" max="100" value={bulkTopK} onChange={(event) => setBulkTopK(Number(event.target.value))} />
                </label>
                {bulkMode === "clip_filtered" && (
                  <label>
                    Max
                    <input type="number" min="1" value={bulkMaxImages} onChange={(event) => setBulkMaxImages(Number(event.target.value))} />
                  </label>
                )}
              </div>
              <button onClick={runBulk} title="Start bulk review job">
                <Sparkles size={17} />
                Start Bulk Review Job
              </button>
            </section>

            <section className="panel">
              <div className="section-title">Jobs</div>
              {bulkJobs.some((job) => ["completed", "failed", "cancelled"].includes(job.status)) && (
                <button className="panel-action" onClick={clearFinishedBulkJobs} title="Remove completed, failed, and cancelled bulk jobs">
                  <Trash2 size={16} />
                  Clear Finished Jobs
                </button>
              )}
              {bulkJobs.length === 0 && <p className="hint">No bulk jobs yet.</p>}
              <div className="results">
                {bulkJobs.map((job) => (
                  <div key={job.id} className={`job-row ${job.id === selectedJobId ? "active" : ""}`}>
                    <button className="job-main" onClick={() => setSelectedJobId(job.id)}>
                      Job #{job.id} {job.result.cancel_requested && job.status === "running" ? "cancelling" : job.status}
                      <small>{job.result.processed ?? 0}/{job.result.total ?? 0} | {job.result.created ?? 0} candidates</small>
                    </button>
                    {["queued", "running"].includes(job.status) && (
                      <button className="icon-only" onClick={() => cancelBulkJob(job)} title={`Cancel Job #${job.id}`}>
                        <X size={16} />
                      </button>
                    )}
                    {!["queued", "running"].includes(job.status) && (
                      <button className="icon-only" onClick={() => deleteBulkJob(job)} title={`Remove Job #${job.id}`}>
                        <Trash2 size={16} />
                      </button>
                    )}
                  </div>
                ))}
              </div>
              {selectedJob?.result.error && <p className="hint error">{selectedJob.result.error}</p>}
            </section>

            <section className="panel">
              <div className="section-title">Pending Review {reviewCandidates.length ? `(${reviewCandidates.length})` : ""}</div>
              {reviewCandidates.length > 0 && (
                <div className="bulk-rename">
                  <input
                    value={bulkRenameText}
                    onChange={(event) => setBulkRenameText(event.target.value)}
                    placeholder="Rename all pending candidates"
                  />
                  <button
                    onMouseDown={() => {
                      suppressReviewRenameBlurRef.current = true;
                    }}
                    onClick={renameAllPendingReviewCandidates}
                    disabled={!bulkRenameText.trim()}
                  >
                    Rename All
                  </button>
                  <button onClick={acceptAllPendingReviewCandidates} disabled={reviewCandidates.length === 0}>
                    Accept All
                  </button>
                </div>
              )}
              {reviewGroups.length === 0 && <p className="hint">No pending candidates for this job.</p>}
              {reviewGroups.map((group) => (
                <div className="review-group" key={group.image.id}>
                  <button className="review-image" onClick={() => navigateToImage(group.image)}>
                    {group.image.file_name}
                    <small>{group.candidates.length} candidates</small>
                  </button>
                  {group.candidates.map((candidate) => (
                    <div className="review-candidate" key={candidate.id}>
                      <div className="review-candidate-main">
                        <span>#{candidate.id} rank {candidate.rank}</span>
                        <input
                          className="inline-name"
                          value={candidate.category_name}
                          onChange={(event) =>
                            setReviewCandidates((items) =>
                              items.map((item) => (item.id === candidate.id ? { ...item, category_name: event.target.value } : item))
                            )
                          }
                          onBlur={(event) => {
                            if (suppressReviewRenameBlurRef.current) return;
                            renameReviewCandidate(candidate, event.target.value);
                          }}
                          onKeyDown={(event) => {
                            if (event.key === "Enter") event.currentTarget.blur();
                          }}
                          title={`Rename review candidate #${candidate.id}`}
                        />
                        <small>{candidate.score == null ? "manual" : candidate.score.toFixed(2)} | {candidate.area}px</small>
                      </div>
                      <div className="review-actions">
                        <button onClick={() => acceptReviewCandidate(candidate.id)}>Accept</button>
                        <button onClick={() => rejectReviewCandidate(candidate.id)}>Reject</button>
                        <button onClick={() => openReviewCandidate(candidate.id)}>Open</button>
                      </div>
                    </div>
                  ))}
                </div>
              ))}
            </section>
          </>
        )}

        {rightTab === "export" && (
          <>
            <section className="panel">
              <div className="section-title">Export Workspace</div>
              <p className="hint">
                {exportLoading
                  ? "Loading annotated folders..."
                  : exportFolders.length > 0
                    ? `${exportAnnotatedCount} annotated images across ${exportFolders.length} folders.`
                    : "Load annotated folders to prepare dataset splits."}
              </p>
              <label className="field-label">
                Export Root
                <input value={exportRoot} onChange={(event) => setExportRoot(event.target.value)} />
              </label>
              <div className="compact-row export-controls">
                <button onClick={() => void loadExportWorkspace()} disabled={exportLoading}>
                  {exportLoading ? "Refreshing..." : "Refresh Folders"}
                </button>
                <button onClick={autoSplitFolders} disabled={exportFolders.length === 0}>
                  Auto Split
                </button>
                <button onClick={combineSelectedFolders} disabled={exportCombineSelection.length === 0}>
                  Combine Selected
                </button>
                <button onClick={runExport} disabled={exportLoading}>
                  <Download size={16} />
                  Export COCO
                </button>
              </div>
              <div className="split-pref-grid">
                <label>
                  Train %
                  <input type="number" min="0" max="100" value={exportTrainPercent} onChange={(event) => setExportTrainPercent(Number(event.target.value))} />
                </label>
                <label>
                  Val %
                  <input type="number" min="0" max="100" value={exportValPercent} onChange={(event) => setExportValPercent(Number(event.target.value))} />
                </label>
                <label>
                  Test %
                  <input type="number" min="0" max="100" value={exportTestPercent} onChange={(event) => setExportTestPercent(Number(event.target.value))} />
                </label>
              </div>
            </section>

            <section className="panel">
              <div className="section-title">Annotated Images by Folder</div>
              {exportFolders.length === 0 && <p className="hint">No annotated folders loaded yet.</p>}
              {exportFolders.map((folder) => (
                <div key={folder.folder} className="review-group export-folder-card">
                  <div className="export-folder-head">
                    <div>
                      <strong>{folder.folder}</strong>
                      <small>
                        {folder.annotatedImages.length} annotated images | {folder.annotationCount} annotations
                      </small>
                    </div>
                    <div className="compact-row">
                      <button onClick={() => splitFolder(folder.folder)}>Split Folder</button>
                      <button onClick={() => setExportSplitPlans((current) => {
                        const next = { ...current };
                        delete next[folder.folder];
                        return next;
                      })} disabled={!exportSplitPlans[folder.folder]}>
                        Clear Split
                      </button>
                    </div>
                  </div>
                  {exportSplitPlans[folder.folder] && (
                    <div className="export-folder-summary">
                      <small>
                        Train {exportSplitPlans[folder.folder].train.length} | Val {exportSplitPlans[folder.folder].val.length} | Test {exportSplitPlans[folder.folder].test.length}
                      </small>
                    </div>
                  )}
                </div>
              ))}
            </section>

            <section className="panel">
              <div className="section-title">Split Buckets</div>
              {exportFolders.length > 0 && (
                <button className="panel-action" onClick={autoSplitFolders}>
                  Split Every Folder
                </button>
              )}
              <div className="split-columns">
                {(["train", "val", "test"] as ExportSplit[]).map((split) => (
                  <div key={split} className="split-column">
                    <strong>{split.toUpperCase()}</strong>
                    <small>{exportBuckets[split].length} folder(s)</small>
                    <div className="split-folder-list">
                      {exportBuckets[split].map((folder) => (
                        <button
                          key={folder.folder}
                          className={exportCombineSelection.includes(folder.folder) ? "split-folder active" : "split-folder"}
                          onClick={() => toggleExportCombineSelection(folder.folder)}
                        >
                          <span>{folder.folder}</span>
                          <small>{folder.images.length}</small>
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </section>

            <section className="panel">
              <div className="section-title">Combined Datasets</div>
              {exportCombinedGroups.length === 0 && <p className="hint">Select folders from the split buckets and combine them here.</p>}
              {exportCombinedGroups.map((group) => (
                <div key={group.id} className="review-group">
                  <strong>{group.name}</strong>
                  <small>{group.folders.length} folder(s)</small>
                  <div className="split-folder-list">
                    {group.folders.map((folder) => (
                      <div key={folder} className="split-folder static">{folder}</div>
                    ))}
                  </div>
                </div>
              ))}
            </section>
          </>
        )}
      </aside>
    </main>
  );
}

function hexToRgba(hex: string, alpha: number) {
  const value = hex.replace("#", "");
  const r = parseInt(value.slice(0, 2), 16);
  const g = parseInt(value.slice(2, 4), 16);
  const b = parseInt(value.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function blobToDataUrl(blob: Blob) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

function loadImage(src: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Could not load mask image."));
    image.src = src;
  });
}

function findNearestPoint(points: Point[], target: Point, radius = 12) {
  let best: number | null = null;
  let bestDistance = radius * radius;
  points.forEach((point, index) => {
    const dx = point.x - target.x;
    const dy = point.y - target.y;
    const distance = dx * dx + dy * dy;
    if (distance <= bestDistance) {
      best = index;
      bestDistance = distance;
    }
  });
  return best;
}

function maskPixels(mask: HTMLImageElement, width: number, height: number) {
  const offscreen = document.createElement("canvas");
  offscreen.width = width;
  offscreen.height = height;
  const off = offscreen.getContext("2d");
  if (!off) return null;
  off.imageSmoothingEnabled = false;
  off.drawImage(mask, 0, 0, width, height);
  return off.getImageData(0, 0, width, height);
}

function isMaskPixel(data: Uint8ClampedArray, index: number) {
  const alpha = data[index + 3];
  const luminance = data[index] + data[index + 1] + data[index + 2];
  return alpha > 24 && luminance > 24;
}

function isMaskPixelAt(pixels: ImageData, x: number, y: number) {
  if (x < 0 || y < 0 || x >= pixels.width || y >= pixels.height) return false;
  return isMaskPixel(pixels.data, (y * pixels.width + x) * 4);
}

function bboxHandles(bbox: number[]) {
  const [x, y, width, height] = bbox;
  const midX = x + width / 2;
  const midY = y + height / 2;
  return [
    { name: "nw" as const, x, y },
    { name: "n" as const, x: midX, y },
    { name: "ne" as const, x: x + width, y },
    { name: "e" as const, x: x + width, y: midY },
    { name: "se" as const, x: x + width, y: y + height },
    { name: "s" as const, x: midX, y: y + height },
    { name: "sw" as const, x, y: y + height },
    { name: "w" as const, x, y: midY }
  ];
}

function resizedBbox(bbox: number[], handle: ResizeHandle, point: Point, imageWidth: number, imageHeight: number) {
  let [x, y, width, height] = bbox;
  let left = x;
  let top = y;
  let right = x + width;
  let bottom = y + height;
  if (handle.includes("w")) left = point.x;
  if (handle.includes("e")) right = point.x;
  if (handle.includes("n")) top = point.y;
  if (handle.includes("s")) bottom = point.y;
  if (right < left) [left, right] = [right, left];
  if (bottom < top) [top, bottom] = [bottom, top];
  left = Math.max(0, Math.min(imageWidth - 2, left));
  top = Math.max(0, Math.min(imageHeight - 2, top));
  right = Math.max(left + 2, Math.min(imageWidth, right));
  bottom = Math.max(top + 2, Math.min(imageHeight, bottom));
  return [left, top, right - left, bottom - top];
}

function transformMaskToBbox(dataUrl: string, fromBbox: number[], toBbox: number[], width: number, height: number) {
  return new Promise<string>((resolve) => {
    const img = new Image();
    img.onload = () => {
      const source = document.createElement("canvas");
      source.width = width;
      source.height = height;
      const sourceCtx = source.getContext("2d");
      const output = document.createElement("canvas");
      output.width = width;
      output.height = height;
      const outputCtx = output.getContext("2d");
      if (!sourceCtx || !outputCtx) {
        resolve(dataUrl);
        return;
      }
      sourceCtx.drawImage(img, 0, 0, width, height);
      outputCtx.imageSmoothingEnabled = false;
      outputCtx.drawImage(
        source,
        fromBbox[0],
        fromBbox[1],
        fromBbox[2],
        fromBbox[3],
        toBbox[0],
        toBbox[1],
        toBbox[2],
        toBbox[3]
      );
      resolve(output.toDataURL("image/png"));
    };
    img.onerror = () => resolve(dataUrl);
    img.src = dataUrl;
  });
}

createRoot(document.getElementById("root")!).render(<App />);
