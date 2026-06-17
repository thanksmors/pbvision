"""Shared engine exceptions."""

from __future__ import annotations


class ModelUnavailable(RuntimeError):
    """A model-backed stage can't run because its deps or weights aren't installed.

    Raised by the lazy ``_ensure_model`` of each detector. The pipeline catches it per-stage
    and degrades gracefully (skips that stage, records a warning) so the engine can be brought
    up one model at a time — e.g. run real player tracking before the court/ball models exist.
    """


class CourtNotFound(RuntimeError):
    """The court detector ran but couldn't localize enough keypoints for a homography.

    Distinct from :class:`ModelUnavailable` (which means the model isn't installed). The
    pipeline treats it the same way for the court stage — skip with a warning — so a clip with
    an awkward camera angle degrades to "no court" instead of crashing the whole run.
    """

