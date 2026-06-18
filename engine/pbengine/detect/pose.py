"""COCO-17 pose keypoint constants — the single source of truth for skeleton topology.

Ultralytics pose models emit keypoints in the fixed COCO ordering below. These constants are pure
data (no torch), so they import cleanly without the ``ml`` extra and are shared by the pose-lift
math (:mod:`pbengine.players.pose3d`) and the pipeline. The frontend mirrors ``COCO_SKELETON`` for
2D line drawing; all *world* geometry is computed server-side.
"""

from __future__ import annotations

# Fixed COCO-17 ordering (index -> name) as emitted by ultralytics pose models.
COCO_KEYPOINTS: tuple[str, ...] = (
    "nose",            # 0
    "left_eye",        # 1
    "right_eye",       # 2
    "left_ear",        # 3
    "right_ear",       # 4
    "left_shoulder",   # 5
    "right_shoulder",  # 6
    "left_elbow",      # 7
    "right_elbow",     # 8
    "left_wrist",      # 9
    "right_wrist",     # 10
    "left_hip",        # 11
    "right_hip",       # 12
    "left_knee",       # 13
    "right_knee",      # 14
    "left_ankle",      # 15
    "right_ankle",     # 16
)
N_KEYPOINTS = len(COCO_KEYPOINTS)

# Convenience index aliases.
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

# Limb / torso / head edges connecting keypoint indices — mirrored in the frontend.
COCO_SKELETON: tuple[tuple[int, int], ...] = (
    (L_ANKLE, L_KNEE), (L_KNEE, L_HIP),         # left leg
    (R_ANKLE, R_KNEE), (R_KNEE, R_HIP),         # right leg
    (L_HIP, R_HIP),                             # pelvis
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),   # torso sides
    (L_SHOULDER, R_SHOULDER),                   # shoulders
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),  # left arm
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),  # right arm
    (NOSE, L_SHOULDER), (NOSE, R_SHOULDER),     # neck/head
)
