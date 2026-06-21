"""Shared ball-detection diagnostics: coverage, inter-detection speed, and gap structure.

Used by both the live pipeline (logged to ``run.log`` on every analysis so a real run is
self-documenting) and ``scripts/debug_ball.py`` (calibration). The whole point is to tell, from a
real clip's log alone, whether fast balls are **missed by the CNN** (low coverage + large gaps where
the ball moves fast) versus **killed downstream** (split into rallies / rejected by the jump gate),
without re-running inference.
"""

from __future__ import annotations

# Gap thresholds mirror the downstream behaviour so the buckets mean something:
#  - <= BRIDGE_DELTA frame-step: trajectory3d.fill_gaps_3d (max_fill_gap=6) physics-bridges it.
#  - > 0.6 s step: rally segmentation starts a new rally.
BRIDGE_DELTA = 6  # frames between consecutive detections that still get interpolated


def classify_gaps(detected_frames, fps: float, bridge_delta: int = BRIDGE_DELTA):
    """Bucket the gaps *between* consecutive detections by how the pipeline treats them.

    Returns ``(buckets, rally_delta)`` where ``buckets`` maps name -> list of ``(start_frame, delta)``
    (delta = frame distance to the next detection). Leading/trailing missing frames are ignored —
    only interior gaps affect arc continuity.
    """
    rally_delta = max(bridge_delta + 1, round(0.6 * fps))  # >0.6 s -> rally split
    fs = sorted({int(f) for f in detected_frames})
    buckets = {"bridged": [], "arc_break": [], "rally_split": []}
    for a, b in zip(fs, fs[1:]):
        d = b - a
        if d <= 1:
            continue
        if d <= bridge_delta:
            buckets["bridged"].append((a, d))
        elif d <= rally_delta:
            buckets["arc_break"].append((a, d))
        else:
            buckets["rally_split"].append((a, d))
    return buckets, rally_delta


def gap_report(detected_frames, n: int, fps: float) -> None:
    """Print the gap-length distribution so scattered short misses (harmless — bridged) are told apart
    from clustered long ones (break the arc / split rallies) without re-running inference."""
    buckets, rally_delta = classify_gaps(detected_frames, fps)

    def _missing(items):  # frames with no detection inside these gaps
        return sum(d - 1 for _, d in items)

    print(f"gap structure (gaps between detections; fps {fps:.0f}, "
          f"bridge<= {BRIDGE_DELTA} frames, rally-split> {rally_delta} frames):", flush=True)
    labels = [("bridged", f"bridged (<= {BRIDGE_DELTA}f, filled by physics)"),
              ("arc_break", f"arc-break ({BRIDGE_DELTA + 1}..{rally_delta}f, visible break)"),
              ("rally_split", f"rally-split (> {rally_delta}f, splits the rally)")]
    for key, label in labels:
        items = buckets[key]
        print(f"  {label}: {len(items)} gaps, {_missing(items)} missing frames", flush=True)
    allgaps = buckets["arc_break"] + buckets["rally_split"]
    if allgaps:
        allgaps.sort(key=lambda g: g[1], reverse=True)
        worst = ", ".join(f"{a / fps:.1f}s (+{d}f/{d / fps:.1f}s)" for a, d in allgaps[:5])
        print(f"  largest breaks at: {worst}", flush=True)
        print("  => scrub the overlay to these timestamps to see if the ball is lost in fast motion.",
              flush=True)
    else:
        print("  => no arc-breaking gaps: all gaps are physics-bridged.", flush=True)


def _percentile(sorted_vals, q: float) -> float:
    """Linear-interpolation percentile (q in 0..100) on an already-sorted list; no numpy dependency."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (q / 100.0) * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def coverage_report(samples, n_frames: int, fps: float, gate_px: float | None = None,
                    court_outliers: int | None = None) -> None:
    """Log ball coverage, inter-detection pixel speed, and gap structure for one analysis run.

    ``samples`` are the post-gate :class:`~pbengine.schema.models.BallSample` (``.frame``, ``.px``).
    A *low coverage with large arc-breaking gaps* is the fast-ball-missed-by-CNN signature; *high
    coverage but many rally-splits* points downstream instead. ``court_outliers`` (when a homography
    was solved) is how many detections had their ground projection discarded as implausibly off-court.
    """
    det = len(samples)
    cov = (det / n_frames) if n_frames else 0.0
    print(f"ball: detected on {det}/{n_frames} frames (coverage {cov * 100:.0f}%)"
          + (f" · jump-gate {gate_px:.0f} px/frame" if gate_px is not None else ""), flush=True)
    if court_outliers is not None:
        print(f"  court-outliers dropped (px kept, position discarded): {court_outliers}/{det}",
              flush=True)

    ss = sorted(samples, key=lambda s: s.frame)
    speeds = []
    for a, b in zip(ss, ss[1:]):
        df = max(1, int(b.frame - a.frame))
        dist = ((b.px[0] - a.px[0]) ** 2 + (b.px[1] - a.px[1]) ** 2) ** 0.5
        speeds.append(dist / df)
    if speeds:
        sp = sorted(speeds)
        gated = f" · over-gate {sum(1 for v in sp if gate_px and v > gate_px)}" if gate_px else ""
        print(f"  inter-detection speed px/frame: median {_percentile(sp, 50):.0f} · "
              f"p90 {_percentile(sp, 90):.0f} · p99 {_percentile(sp, 99):.0f} · max {sp[-1]:.0f}"
              + gated, flush=True)
    gap_report([s.frame for s in ss], n_frames, fps)
    if cov < 0.2:
        print("  => low coverage: the WASB CNN is missing the ball on most frames (it sees a single "
              "512x288 downscaled pass; a fast ball shrinks to a few px). Likely needs a high-res "
              "crop / motion-cue fallback, not a threshold tweak.", flush=True)
