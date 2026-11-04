import { useEffect, useRef, useState, type ReactNode } from "react";

/** Nearest source frame for a media time, using the server's exact PTS table (t relative to the first frame). */
export function frameAt(times: number[], t: number): number {
  if (!times.length) return 0;
  let lo = 0;
  let hi = times.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (times[mid] <= t + 1e-4) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

/** HTML5 video with a frame/time readout derived from the exact frame table. The browser's media clock is
 *  approximate for the frame number (labelled "≈"); the frame inspector shows exact decoded frames. */
export default function VideoPlayer({
  src, times, label, overlay, onFrame, tag,
}: {
  src: string;
  times?: number[];
  label?: string;
  tag?: ReactNode;
  overlay?: (frame: number) => ReactNode;
  onFrame?: (frame: number, t: number) => void;
}) {
  const ref = useRef<HTMLVideoElement>(null);
  const [t, setT] = useState(0);
  const [failed, setFailed] = useState(false);
  const frame = times ? frameAt(times, t) : null;

  useEffect(() => {
    const v = ref.current as (HTMLVideoElement & { requestVideoFrameCallback?: (cb: (n: number, m: { mediaTime: number }) => void) => number; cancelVideoFrameCallback?: (h: number) => void }) | null;
    if (!v) return;
    let handle = 0;
    if (v.requestVideoFrameCallback) {
      const cb = (_: number, m: { mediaTime: number }) => {
        setT(m.mediaTime);
        handle = v.requestVideoFrameCallback!(cb);
      };
      handle = v.requestVideoFrameCallback(cb);
      const onSeek = () => setT(v.currentTime);
      v.addEventListener("seeked", onSeek);
      return () => {
        v.cancelVideoFrameCallback?.(handle);
        v.removeEventListener("seeked", onSeek);
      };
    }
    const on = () => setT(v.currentTime);
    v.addEventListener("timeupdate", on);
    v.addEventListener("seeked", on);
    return () => {
      v.removeEventListener("timeupdate", on);
      v.removeEventListener("seeked", on);
    };
  }, [src]);

  useEffect(() => {
    if (frame !== null) onFrame?.(frame, t);
  }, [frame, t, onFrame]);

  const step = (d: number) => {
    const v = ref.current;
    if (!v || !times || frame === null) return;
    v.pause();
    const i = Math.max(0, Math.min(times.length - 1, frame + d));
    // aim for the middle of the target frame's display interval
    const next = i + 1 < times.length ? times[i + 1] : times[i] + 1 / 30;
    v.currentTime = (times[i] + next) / 2;
  };

  return (
    <div>
      <div className="media">
        {failed ? (
          <div className="empty">This browser cannot play the video codec. The analysis is unaffected; use the frame inspector.</div>
        ) : (
          <video ref={ref} src={src} controls playsInline muted preload="metadata" onError={() => setFailed(true)} />
        )}
        {overlay && frame !== null && !failed && overlay(frame)}
        {tag && <span className="tag">{tag}</span>}
      </div>
      <div className="transport">
        {times && (
          <>
            <button className="btn sm" onClick={() => step(-1)} aria-label="previous frame">
              ◀ frame
            </button>
            <button className="btn sm" onClick={() => step(1)} aria-label="next frame">
              frame ▶
            </button>
          </>
        )}
        <span className="grow" />
        <span className="timecode">
          {label ? `${label} · ` : ""}t = {t.toFixed(3)} s{frame !== null ? ` · frame ≈ ${frame}` : ""}
        </span>
      </div>
    </div>
  );
}
