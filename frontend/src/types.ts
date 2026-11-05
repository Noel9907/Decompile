// Shapes returned by the Decompile backend (decompile_web). Only fields the UI reads are typed.

export type StageStatus = "pending" | "running" | "done" | "failed" | "skipped" | "interrupted";

export interface Stage {
  key: string;
  title: string;
  status: StageStatus;
  detail: Record<string, unknown> | null;
  started_utc: string | null;
  finished_utc: string | null;
}

export interface Cost {
  frames_in_video: number;
  billing_minimum_frames: number;
  expected_billed_frames: number;
  usd_per_1000_frames: number;
  expected_cost_usd: number;
  pricing_source: string;
  caveat: string;
}

export interface Plan {
  sha256: string;
  created_utc: string;
  prompt: Record<string, unknown> & { kind: string; text?: string; input_text?: string };
  cost: Cost;
  media: { source_frames: number; displayed_width: number; displayed_height: number; first_t_s: number; last_t_s: number };
  warnings: string[];
  expected_processed_frames: number;
  request_preview: unknown;
  live_call_made: boolean;
}

export interface SamAttempt {
  n: number;
  status: string;
  started_utc: string;
  finished_utc?: string;
  usage?: Record<string, number> | null;
  actual_billed_frames?: number | null;
  actual_cost_usd?: number | null;
  request_sent?: boolean;
  error?: string;
  error_kind?: string;
  billing_note?: string;
}

export interface SamObjectSummary {
  n_frames: number;
  first_frame: number;
  last_frame: number;
  n_with_mask: number;
  median_box_area_px: number;
  min_confidence: number | null;
  median_confidence: number | null;
}

export interface SamState {
  status: string;
  source?: "live" | "imported";
  attempts: SamAttempt[];
  warnings: string[];
  result: null | {
    path: string;
    sha256: string;
    model: string;
    prompt: Record<string, unknown>;
    outcome: { status: string; reason?: string | null };
    usage: Record<string, number> | null;
    summary: { n_objects: number; n_frames_with_records: number; objects: Record<string, SamObjectSummary> };
    provenance: Record<string, unknown>;
  };
}

export interface ReconParams {
  sam_object: string;
  start_frame: number;
  end_frame: number | null;
  physics_model: "projective" | "image_space";
  outline_model: string;
  hfov_deg: number;
  min_confidence: number;
}

export interface Reconstruction {
  id: string;
  params: ReconParams;
  status: "queued" | "running" | "completed" | "failed" | "interrupted";
  created_utc: string;
  finished_utc: string | null;
  error: string | null;
  outcome: "scene_program" | "measurements_only" | "stopped" | "error" | null;
  stages: Record<string, Stage>;
}

export interface Check {
  ok: boolean;
  label: string;
  detail: string;
  level: "required" | "advisory";
}

export interface Run {
  id: string;
  created_utc: string;
  updated_utc: string;
  original_filename: string;
  upload_size_bytes: number;
  provenance: "real" | "synthetic";
  state: string;
  video: null | {
    sha256: string;
    size_bytes: number;
    codec: string | null;
    width: number | null;
    height: number | null;
    duration_s: number | null;
    nominal_fps: string | null;
    decoded_frame_count: number | null;
    frame_count?: number;
    display_width?: number;
    display_height?: number;
    time_base?: string;
    synthesized_timestamp_count?: number;
  };
  suitability: null | { suitable: boolean; checks: Check[]; warnings: string[]; scope: string };
  rejection_reasons: string[];
  stages: Record<string, Stage>;
  plan: Plan | null;
  sam: SamState;
  reconstructions: Reconstruction[];
  job: null | { kind: string; status: string; error: string | null; details?: string[] };
}

export interface RunListItem {
  id: string;
  created_utc: string;
  original_filename: string;
  state: string;
  provenance: string;
  frames: number | null;
  reconstructions: number;
  has_program: boolean;
}

export interface Health {
  ok: boolean;
  version: string;
  live_sam_enabled: boolean;
  api_key_configured: boolean;
  max_upload_bytes: number;
  spark_available?: boolean;
  max_sam_cost_usd: number;
  pricing: { usd_per_1000_frames: number; billing_minimum_frames: number; source: string };
  max_window_frames: number;
  limitations: string[];
}

export interface FrameRow {
  frame_index: number;
  t_s: number;
  t_exact: { num: number; den: number };
  status: "measured" | "uncertain" | "missing";
  reason: string | null;
  x: number | null;
  y: number | null;
  sigma_x: number | null;
  sigma_y: number | null;
  diameter_px: number | null;
  sigma_diameter_px: number | null;
  confidence: number | null;
  mask_area_px: number | null;
  mask_centroid: [number, number] | null;
  iou_outline_mask: number | null;
  centre_offset_from_mask_px: number | null;
  outline: null | { semi_major_px: number; semi_minor_px: number; angle_deg: number; [k: string]: unknown };
}

export interface SimEvent {
  t_s: number;
  x_px: number;
  y_px: number;
  resting?: boolean;
}

export interface Sim {
  t: number[];
  x: number[];
  y: number[];
  d?: number[];
  events: SimEvent[];
  termination: string | null;
  valid_until_s?: number;
  validity_note?: string;
}

/** Engine samples after valid_until_s (projective body at rest) are not physically meaningful: drop them. */
export function validPart(s: Sim): Sim {
  if (s.valid_until_s === undefined) return s;
  const n = s.t.findIndex((t) => t > s.valid_until_s! + 1e-9);
  if (n < 0) return s;
  return { ...s, t: s.t.slice(0, n), x: s.x.slice(0, n), y: s.y.slice(0, n), d: s.d?.slice(0, n) };
}

export interface EngineReport {
  label: string;
  video: { filename: string; frames: number; displayed_size: [number, number] };
  sam: { model: string; prompt: Record<string, unknown>; object_id: string; objects_in_result: string[]; usage: unknown; summary: SamObjectSummary | null };
  clip: { start_frame: number; end_frame_exclusive: number; frames: number };
  measurement: {
    method: string;
    status_counts: { measured: number; uncertain: number; missing: number };
    median_sigma_px: number | null;
    median_diameter_px: number | null;
    median_abs_mask_centroid_offset_px: number | null;
  };
  physics: Record<string, any>;
  reconstruction_consistency?: { rms_px: number; rms_normalised: number; n: number; events: SimEvent[]; note: string };
  claims: string[];
  artifacts: string[];
}

export interface Analysis {
  recon_id: string;
  frames: FrameRow[];
  report: EngineReport;
  program: Record<string, any> | null;
  program_parameters: Record<string, any> | null;
  fitted: (Sim & { frame_index: number[] }) | null;
  fitted_dense: Sim | null;
  physics_status: { status: "scene_program" | "no_program"; reasons?: string[]; adequacy?: string } | null;
}

export interface PerceptionObject {
  id: string;
  box: [number, number, number, number];
  box_confidence: number | null;
  mask_confidence: number | null;
  has_mask: boolean;
  contours: [number, number][][];
}

export interface Perception {
  selected_object: string;
  width: number;
  height: number;
  frames: Record<string, PerceptionObject[]>;
  note: string;
}

export interface Counterfactual {
  id: string;
  recon_id: string;
  label: string;
  parameters: { restitution: number | null; gravity_scale: number | null; extend_s: number; velocity_scale?: number; remove_floor?: boolean };
  fitted_parameters: Record<string, any>;
  counterfactual_parameters: Record<string, any>;
  frames: { frame_index: number[]; fitted: Sim; counterfactual: Sim };
  dense: { fitted: Sim; counterfactual: Sim };
  observed_until_s: number;
  source?: { display: string; exact: string };
}

export interface FitReview {
  headline: string;
  explanation: string;
  material_guess: string;
  plausibility: string;
  critique: string[];
  confidence: "high" | "medium" | "low";
  model: string | null;
  latency_s: number;
}

export interface AskResult {
  interpretation: {
    question: string;
    supported: boolean;
    restitution: number | null;
    gravity_scale: number | null;
    velocity_scale?: number | null;
    remove_floor?: boolean;
    extend_s: number;
    headline: string;
    reasoning: string;
    assumptions: string[];
    unsupported_reason: string | null;
    model: string | null;
    latency_s: number;
  };
  counterfactual: Counterfactual | null;
}

export interface Artifact {
  path: string;
  category: string;
  label: string;
  size_bytes: number;
  scope: string | null;
  media_type: string;
}

export interface RunEvent {
  seq: number;
  t: string;
  type: string;
  [k: string]: unknown;
}
