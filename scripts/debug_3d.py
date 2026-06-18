"""Reconstruct the ball's 3D trajectory (feet) and report height + speed on a real clip.

Chains: court homography -> metric camera (pbengine.ball.camera) -> WASB ball track ->
gravity-constrained 3D lift (pbengine.ball.trajectory3d). Prints the recovered focal length,
camera health, and per-segment peak height / top speed so you can sanity-check the 3D before
trusting it. Single-camera 3D is approximate; the corner-reprojection error is your reliability
gauge (small = trustworthy).

    # automatic court detection:
    python scripts/debug_3d.py clip.mp4
    # or hand the calibrated corners (recommended on pickleball, where auto-court may not transfer):
    python scripts/debug_3d.py clip.mp4 --court-corners corners.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

import numpy as np  # noqa: E402

from pbengine.ball.camera import recover_camera  # noqa: E402
from pbengine.ball.tracker import BallTracker  # noqa: E402
from pbengine.ball.trajectory3d import reconstruct_3d  # noqa: E402
from pbengine.bounce.heuristic import detect_bounces  # noqa: E402
from pbengine.io.video import iter_frames, probe  # noqa: E402


def _homography(video: str, corners_path: str | None) -> np.ndarray:
    if corners_path:
        from pbengine.court.detector import ManualCourtDetector, load_corners

        return ManualCourtDetector(load_corners(corners_path)).solve()
    from pbengine.court.detector import CourtDetector
    from pbengine.court.homography import homography_from_named_points

    det = CourtDetector()
    for _idx, frame in iter_frames(video):
        named = det.detect(frame)
        if len(named) < 4:
            raise SystemExit(
                f"auto court detection found only {len(named)}/4 corners. "
                "Calibrate manually and pass --court-corners corners.json"
            )
        return homography_from_named_points(named)
    raise SystemExit("no frames")


def main(video: str, corners_path: str | None) -> int:
    meta = probe(video)
    print(f"video: {meta.width}x{meta.height} @ {meta.fps:.1f}fps, {meta.frames} frames")

    cam = recover_camera(_homography(video, corners_path), meta.width, meta.height)
    print(f"camera: focal~{cam.focal_px:.0f}px | corner reprojection error "
          f"{cam.reprojection_error_px:.1f}px "
          f"({'reliable' if cam.reprojection_error_px < 25 else 'POOR — 3D suspect'})")

    print("tracking ball (slow on CPU)...")
    samples = BallTracker().track(video)
    bounces = detect_bounces(samples)
    out = reconstruct_3d(samples, bounces, cam, meta.fps)

    lifted = [s for s in out if s.world_ft is not None]
    print(f"ball detected: {len(samples)} frames | lifted to 3D: {len(lifted)} | "
          f"bounces: {len(bounces)}")
    if not lifted:
        print("=> No 3D recovered. Check that the ball track and court calibration overlap.")
        return 0

    heights = [s.world_ft[2] for s in lifted]
    speeds = [s.speed_mph for s in lifted if s.speed_mph is not None]
    print(f"peak height: {max(heights):.1f} ft | top speed: {max(speeds):.0f} mph "
          f"| median speed: {sorted(speeds)[len(speeds) // 2]:.0f} mph")
    print("samples (frame: X,Y,Z ft @ mph):")
    for s in lifted[:: max(1, len(lifted) // 8)][:8]:
        x, y, z = s.world_ft
        print(f"  {s.frame:5d}: {x:5.1f},{y:5.1f},{z:4.1f}  @ {s.speed_mph:3.0f} mph")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Reconstruct + report the ball's 3D trajectory.")
    ap.add_argument("video")
    ap.add_argument("--court-corners", help="JSON of 4 clicked court corners (manual calibration)")
    args = ap.parse_args()
    raise SystemExit(main(args.video, args.court_corners))
