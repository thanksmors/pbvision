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
    # also write a rotatable 3D viewer you can open in a browser (drag to orbit):
    python scripts/debug_3d.py clip.mp4 --court-corners corners.json --html ball_3d.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

import numpy as np  # noqa: E402

from pbengine.ball.camera import recover_camera  # noqa: E402
from pbengine.ball.tracker import BallTracker  # noqa: E402
from pbengine.ball.trajectory3d import WIDTH_FT, LENGTH_FT, fill_gaps_3d, reconstruct_3d  # noqa: E402
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


_VIEWER_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>pbvision 3D ball — __TITLE__</title>
<style>
 html,body{margin:0;height:100%;overflow:hidden;background:#0d1117;font-family:system-ui,sans-serif}
 #info{position:absolute;top:10px;left:12px;color:#c9d1d9;font-size:13px;line-height:1.5;
       text-shadow:0 1px 2px #000;pointer-events:none}
 #info b{color:#fff} .k{display:inline-block;width:11px;height:11px;border-radius:2px;
       vertical-align:middle;margin-right:4px}
</style></head><body>
<div id="info">
 <b>pbvision 3D ball</b> &nbsp;·&nbsp; drag = rotate, scroll = zoom, right-drag = pan<br>
 <span class="k" style="background:#3b82f6"></span>slow
 <span class="k" style="background:#ef4444"></span>fast &nbsp;
 <span class="k" style="background:#9ca3af"></span>interpolated &nbsp;
 <span class="k" style="background:#fbbf24"></span>bounce<br>
 <span id="stats"></span>
</div>
<script type="importmap">
{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js",
 "three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
const D = __DATA__;
const W = D.width, L = D.length;                       // court ft
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(devicePixelRatio); renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x0d1117);
const cam = new THREE.PerspectiveCamera(50, innerWidth/innerHeight, 0.1, 1000);
cam.position.set(L*0.9, L*0.7, L*0.9); scene.add(cam);
const controls = new OrbitControls(cam, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.AmbientLight(0xffffff, 0.9));
// world (x across width, y along length, z up) -> three (x, z, y), centered on the court.
const P = (x,y,z)=>new THREE.Vector3(x - W/2, z, y - L/2);
function line(pts, color, opacity=1){
  const g = new THREE.BufferGeometry().setFromPoints(pts);
  const m = new THREE.LineBasicMaterial({color, transparent:opacity<1, opacity});
  scene.add(new THREE.Line(g, m));
}
// Court outline, net, kitchen (7 ft from net), centre line.
line([P(0,0,0),P(W,0,0),P(W,L,0),P(0,L,0),P(0,0,0)], 0x6e7681);
line([P(0,L/2,0),P(W,L/2,0)], 0x8b949e);              // net base
line([P(0,L/2,2.83),P(W,L/2,2.83)], 0x8b949e, 0.7);   // net top (~34in)
[P(0,L/2,0),P(0,L/2,2.83),P(W,L/2,0),P(W,L/2,2.83)].reduce((a,p,i)=>{
  if(i%2) line([a,p],0x8b949e,0.5); return p;});
line([P(0,L/2-7,0),P(W,L/2-7,0)], 0x484f58);
line([P(0,L/2+7,0),P(W,L/2+7,0)], 0x484f58);
line([P(W/2,0,0),P(W/2,L/2-7,0)], 0x484f58);
line([P(W/2,L/2+7,0),P(W/2,L,0)], 0x484f58);
// Trajectory: continuous dim line + per-sample spheres coloured by speed.
const cold = new THREE.Color(0x3b82f6), hot = new THREE.Color(0xef4444);
const path = D.pts.filter(p=>p.z!=null).map(p=>P(p.x,p.y,p.z));
if (path.length>1) line(path, 0x30363d, 0.6);
const sph = new THREE.SphereGeometry(0.22, 12, 12);
for (const p of D.pts){
  if (p.z==null) continue;
  let col;
  if (p.interp) col = new THREE.Color(0x9ca3af);
  else { const t = Math.min(1, (p.mph||0)/D.maxmph); col = cold.clone().lerp(hot, t); }
  const mat = new THREE.MeshBasicMaterial({color:col, transparent:p.interp, opacity:p.interp?0.5:1});
  const s = new THREE.Mesh(p.interp ? new THREE.SphereGeometry(0.14,8,8) : sph, mat);
  s.position.copy(P(p.x,p.y,p.z)); scene.add(s);
}
for (const b of D.bounces){
  const m = new THREE.Mesh(new THREE.SphereGeometry(0.3,12,12),
                           new THREE.MeshBasicMaterial({color:0xfbbf24}));
  m.position.copy(P(b.x,b.y,0.02)); scene.add(m);
}
document.getElementById('stats').textContent = D.stats;
controls.target.set(0, 2, 0);
addEventListener('resize', ()=>{cam.aspect=innerWidth/innerHeight; cam.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);});
(function loop(){requestAnimationFrame(loop); controls.update(); renderer.render(scene, cam);})();
</script></body></html>"""


def _write_viewer(path: str, samples, bounces, stats: str, title: str) -> None:
    pts = [
        {
            "f": s.frame,
            "x": round(s.world_ft[0], 3),
            "y": round(s.world_ft[1], 3),
            "z": round(s.world_ft[2], 3),
            "mph": round(s.speed_mph, 1) if s.speed_mph is not None else None,
            "interp": bool(getattr(s, "interpolated", False)),
        }
        for s in samples
        if s.world_ft is not None
    ]
    maxmph = max((p["mph"] for p in pts if p["mph"]), default=1.0) or 1.0
    data = {
        "width": WIDTH_FT,
        "length": LENGTH_FT,
        "maxmph": maxmph,
        "stats": stats,
        "pts": pts,
        "bounces": [{"x": round(b.court_xy[0] * WIDTH_FT, 3),
                     "y": round(b.court_xy[1] * LENGTH_FT, 3)}
                    for b in bounces if getattr(b, "court_xy", None) is not None],
    }
    html = (_VIEWER_HTML.replace("__DATA__", json.dumps(data))
            .replace("__TITLE__", title))
    Path(path).write_text(html)
    print(f"=> wrote {path} ({len(pts)} 3D points). Open it in a browser and drag to rotate.")


def main(video: str, corners_path: str | None, html_path: str | None = None) -> int:
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

    # Stats come from MEASURED frames only (interpolation must not set a speed/height record).
    heights = [s.world_ft[2] for s in lifted]
    speeds = [s.speed_mph for s in lifted if s.speed_mph is not None]
    peak_h, top_v = max(heights), max(speeds)
    med_v = sorted(speeds)[len(speeds) // 2]
    stats = (f"detected {len(lifted)} · peak {peak_h:.1f} ft · "
             f"top {top_v:.0f} mph · median {med_v:.0f} mph")
    print(f"peak height: {peak_h:.1f} ft | top speed: {top_v:.0f} mph | median speed: {med_v:.0f} mph")
    print("samples (frame: X,Y,Z ft @ mph):")
    for s in lifted[:: max(1, len(lifted) // 8)][:8]:
        x, y, z = s.world_ft
        print(f"  {s.frame:5d}: {x:5.1f},{y:5.1f},{z:4.1f}  @ {s.speed_mph:3.0f} mph")

    if html_path:
        # Gap-fill so the rendered track is continuous; filled points render dim/smaller and are
        # excluded from the stats above. This is what the production pipeline shows.
        merged = fill_gaps_3d(out, bounces, cam, meta.fps)
        n_fill = sum(1 for s in merged if getattr(s, "interpolated", False) and s.world_ft)
        print(f"gap-fill: +{n_fill} interpolated 3D points for a continuous track")
        _write_viewer(html_path, merged, bounces, stats, Path(video).name)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Reconstruct + report the ball's 3D trajectory.")
    ap.add_argument("video")
    ap.add_argument("--court-corners", help="JSON of 4 clicked court corners (manual calibration)")
    ap.add_argument("--html", nargs="?", const="ball_3d.html", default=None,
                    help="write a rotatable 3D viewer (three.js) to this path (default ball_3d.html)")
    args = ap.parse_args()
    raise SystemExit(main(args.video, args.court_corners, args.html))
