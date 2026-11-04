import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { parseHash } from "../App";
import { frameAt } from "../components/VideoPlayer";
import { validateLocalFile } from "../pages/Home";
import PhysicsTab from "../pages/PhysicsTab";
import { ConfirmModal } from "../pages/ProcessTab";
import SimulationTab from "../pages/SimulationTab";
import type { Analysis, Counterfactual, Plan, Run, Sim } from "../types";
import { validPart } from "../types";

// ------------------------------------------------------------------ fixtures

const sim = (n: number, extra: Partial<Sim> = {}): Sim => ({
  t: Array.from({ length: n }, (_, i) => 1 + i * 0.01),
  x: Array.from({ length: n }, (_, i) => 100 + i),
  y: Array.from({ length: n }, (_, i) => 200 + i),
  events: [{ t_s: 1.05, x_px: 105, y_px: 205 }],
  termination: "end_of_horizon",
  ...extra,
});

const run = {
  id: "web_20261105T000000Z_0badf00d",
  original_filename: "bounce.mp4",
  provenance: "real",
  state: "reconstructed",
  video: { size_bytes: 1000, sha256: "a".repeat(64), frame_count: 30 },
  sam: { status: "completed", attempts: [], warnings: [], result: null },
  reconstructions: [],
  stages: {},
} as unknown as Run;

const plan: Plan = {
  sha256: "f".repeat(64),
  created_utc: "",
  prompt: { kind: "concept", text: "ball" },
  cost: { frames_in_video: 30, billing_minimum_frames: 50, expected_billed_frames: 50, usd_per_1000_frames: 0.2, expected_cost_usd: 0.01, pricing_source: "doc", caveat: "c" },
  media: { source_frames: 30, displayed_width: 64, displayed_height: 48, first_t_s: 0, last_t_s: 1 },
  warnings: [],
  expected_processed_frames: 30,
  request_preview: {},
  live_call_made: false,
};

function analysis(withProgram: boolean): Analysis {
  return {
    recon_id: "r1",
    frames: [
      { frame_index: 0, t_s: 1, t_exact: { num: 1, den: 1 }, status: "measured", reason: null, x: 100, y: 200, sigma_x: 0.5, sigma_y: 0.5, diameter_px: 20, sigma_diameter_px: 0.3, confidence: 0.9, mask_area_px: 300, mask_centroid: [100, 200], iou_outline_mask: 0.9, centre_offset_from_mask_px: 0.2, outline: null },
      { frame_index: 1, t_s: 1.033, t_exact: { num: 31, den: 30 }, status: "uncertain", reason: "edge fit rejected", x: null, y: null, sigma_x: null, sigma_y: null, diameter_px: null, sigma_diameter_px: null, confidence: 0.4, mask_area_px: 280, mask_centroid: null, iou_outline_mask: null, centre_offset_from_mask_px: null, outline: null },
    ],
    report: {
      label: "REAL VIDEO",
      video: { filename: "bounce.mp4", frames: 30, displayed_size: [64, 48] },
      sam: { model: "sam-3.1", prompt: {}, object_id: "3", objects_in_result: ["3"], usage: null, summary: null },
      clip: { start_frame: 0, end_frame_exclusive: 2, frames: 2 },
      measurement: { method: "m", status_counts: { measured: 1, uncertain: 1, missing: 0 }, median_sigma_px: 0.5, median_diameter_px: 20, median_abs_mask_centroid_offset_px: 0.2 },
      physics: withProgram ? { model: "image_space", restitution: 0.65, program_sha256: "abc", kinematics: { adequacy: "adequate", selected_model: "one_contact", contact: { t_s: 1.05, between_frames: [0, 1] } } } : { model: "image_space", program: null, reasons: ["only 1 measured frames"] },
      claims: ["fitted, not predicted"],
      artifacts: [],
    },
    program: withProgram ? { provenance: {} } : null,
    program_parameters: withProgram ? { model: "image_space", restitution: 0.65, acceleration_px_s2: [0, 1200], gravity_magnitude_px_s2: 1200, initial_position_px: [1, 2], initial_velocity_px_s: [3, 4], t0_s: 1 } : null,
    fitted: null,
    fitted_dense: null,
    physics_status: withProgram ? { status: "scene_program", adequacy: "adequate" } : { status: "no_program", reasons: ["only 1 measured frames"] },
  };
}

const cfResponse = (e: number): Counterfactual => ({
  id: "cf_0123456789abcdef",
  recon_id: "r1",
  label: "HYPOTHETICAL",
  parameters: { restitution: e, gravity_scale: null, extend_s: 0.6 },
  fitted_parameters: { model: "image_space", restitution: 0.65, gravity_magnitude_px_s2: 1200, acceleration_px_s2: [0, 1200] },
  counterfactual_parameters: { model: "image_space", restitution: e, gravity_magnitude_px_s2: 1200, acceleration_px_s2: [0, 1200] },
  frames: { frame_index: [0, 1], fitted: sim(2), counterfactual: sim(2) },
  dense: { fitted: sim(50), counterfactual: sim(50) },
  observed_until_s: 1.2,
});

let fetchMock: ReturnType<typeof vi.fn>;
beforeEach(() => {
  fetchMock = vi.fn();
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => vi.unstubAllGlobals());

const ok = (body: unknown) => Promise.resolve(new Response(JSON.stringify(body), { status: 200 }));

// ------------------------------------------------------------------ pure helpers

describe("upload validation", () => {
  it("accepts MP4 within the limit and rejects other formats, empty and oversize files", () => {
    expect(validateLocalFile(new File(["x"], "clip.mp4"), 10)).toBeNull();
    expect(validateLocalFile(new File(["x"], "clip.MOV"), 10)).toMatch(/not an MP4/);
    expect(validateLocalFile(new File([], "clip.mp4"), 10)).toMatch(/empty/);
    expect(validateLocalFile(new File(["x".repeat(20)], "clip.mp4"), 10)).toMatch(/accepts up to/);
  });
});

describe("routing keeps the run in the URL", () => {
  it("parses run and tab, defaults to processing", () => {
    expect(parseHash("#/runs/web_1/simulation")).toEqual({ page: "run", id: "web_1", tab: "simulation" });
    expect(parseHash("#/runs/web_1")).toEqual({ page: "run", id: "web_1", tab: "process" });
    expect(parseHash("")).toEqual({ page: "home" });
  });
});

describe("frame lookup from the exact PTS table", () => {
  it("returns the last frame whose timestamp is <= t", () => {
    const ts = [0, 0.0333, 0.0667, 0.1];
    expect(frameAt(ts, 0)).toBe(0);
    expect(frameAt(ts, 0.05)).toBe(1);
    expect(frameAt(ts, 0.0667)).toBe(2);
    expect(frameAt(ts, 5)).toBe(3);
  });
});

describe("post-rest samples", () => {
  it("drops simulator samples after valid_until_s and keeps others untouched", () => {
    const s = sim(10, { valid_until_s: 1.045 });
    expect(validPart(s).t).toHaveLength(5);
    expect(validPart(sim(10)).t).toHaveLength(10);
  });
});

// ------------------------------------------------------------------ paid-call confirmation

describe("SAM confirmation dialog", () => {
  it("requires the explicit checkbox and sends the exact plan hash and cost", async () => {
    fetchMock.mockReturnValue(ok(run));
    const onConfirmed = vi.fn();
    render(<ConfirmModal plan={plan} run={run} onCancel={() => {}} onConfirmed={onConfirmed} />);
    expect(screen.getByText("$0.010", { selector: ".value" })).toBeInTheDocument();
    const go = screen.getByTestId("confirm-sam");
    expect(go).toBeDisabled();
    fireEvent.click(go);
    expect(fetchMock).not.toHaveBeenCalled();
    fireEvent.click(screen.getByTestId("ack"));
    expect(go).toBeEnabled();
    fireEvent.click(go);
    await waitFor(() => expect(onConfirmed).toHaveBeenCalled());
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`/api/runs/${run.id}/sam/confirm`);
    expect(JSON.parse(init.body)).toEqual({ confirm: true, plan_sha256: plan.sha256, acknowledged_cost_usd: 0.01 });
  });

  it("shows the server's refusal and stays open", async () => {
    fetchMock.mockReturnValue(Promise.resolve(new Response(JSON.stringify({ detail: "Live SAM requests are disabled" }), { status: 400 })));
    const onConfirmed = vi.fn();
    render(<ConfirmModal plan={plan} run={run} onCancel={() => {}} onConfirmed={onConfirmed} />);
    fireEvent.click(screen.getByTestId("ack"));
    fireEvent.click(screen.getByTestId("confirm-sam"));
    expect(await screen.findByText(/disabled/)).toBeInTheDocument();
    expect(onConfirmed).not.toHaveBeenCalled();
  });
});

// ------------------------------------------------------------------ missing scene program

describe("missing scene program", () => {
  it("simulation tab explains instead of simulating, and makes no request", () => {
    render(<SimulationTab run={run} rid="r1" analysis={analysis(false)} />);
    expect(screen.getByText(/No executable scene program/)).toBeInTheDocument();
    expect(screen.getByText(/only 1 measured frames/)).toBeInTheDocument();
    expect(screen.queryByLabelText("restitution")).toBeNull();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("physics tab reports the stop and shows no fabricated parameters", () => {
    render(<PhysicsTab run={run} analysis={analysis(false)} />);
    expect(screen.getByText(/Reconstruction stopped before a scene program/)).toBeInTheDocument();
    expect(screen.queryByText("Restitution e")).toBeNull();
  });
});

// ------------------------------------------------------------------ counterfactual updates

describe("counterfactual controls", () => {
  it("re-runs the server simulator with the new restitution and labels the result hypothetical", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    fetchMock.mockImplementation((_url: string, init: RequestInit) => ok(cfResponse(JSON.parse(String(init.body)).restitution)));
    render(<SimulationTab run={run} rid="r1" analysis={analysis(true)} />);
    await act(async () => {
      vi.advanceTimersByTime(300);
    });
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    fireEvent.change(screen.getByLabelText("restitution"), { target: { value: "0.9" } });
    await act(async () => {
      vi.advanceTimersByTime(300);
    });
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    const [url, init] = fetchMock.mock.calls[1];
    expect(url).toBe(`/api/runs/${run.id}/reconstructions/r1/counterfactuals`);
    expect(JSON.parse(init.body)).toMatchObject({ restitution: 0.9, gravity_scale: null });
    expect(await screen.findByText("Fitted vs counterfactual")).toBeInTheDocument();
    expect(screen.getAllByText(/hypothetical/i).length).toBeGreaterThan(0);
    vi.useRealTimers();
  });
});
