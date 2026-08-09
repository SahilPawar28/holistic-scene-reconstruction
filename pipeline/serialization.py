"""
Making pipeline statistics safe to send over JSON.

This lives in the package rather than in the notebook on purpose. The Colab
notebook is updated by re-opening it from GitHub, whereas `pipeline/` is
updated by the `git pull` in cell 2 — so anything defined in a notebook cell
silently keeps running an old version until the notebook itself is
reloaded. Logic that needs to be fixable without that round trip belongs
here.
"""

from __future__ import annotations

import numpy as np


def json_safe(value):
    """Recursively strip NaN/Inf and numpy scalars from a stats tree.

    JSON has no representation for NaN, and `json.dumps` raises rather than
    emitting one — so a single unmeasurable statistic anywhere in the tree
    takes down the entire response and the caller gets no scene at all, only
    a traceback.

    Plenty of these statistics are legitimately NaN: depth relief when the
    ring around a mask has too few valid pixels, RMS error on a fit that
    never converged, semantic confidence when labelling was skipped. They
    are converted to null once, centrally, rather than guarded at every site
    that writes one — the failure mode of missing a single site is losing
    the whole reconstruction.
    """
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    return value
