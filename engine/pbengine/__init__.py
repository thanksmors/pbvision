"""pbengine — pickleball video analysis engine.

Importable and runnable independently of the API. The pure-logic stages (schema, court
geometry, rally/serve/winner heuristics, Kalman smoothing) depend only on the core
requirements; model-backed stages load their heavy dependencies lazily.
"""

__version__ = "0.1.0"
