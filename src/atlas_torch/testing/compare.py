"""Comparison helpers that report max/rms relative error (the scoreboard's error metrics)."""
from __future__ import annotations

import numpy as np
import torch


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def rel_err(a, b, floor: float = 1e-300) -> np.ndarray:
    a, b = to_np(a).astype(np.float64), to_np(b).astype(np.float64)
    return np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), floor)


def err_stats(a, b, floor: float = 1e-300) -> dict:
    r = rel_err(a, b, floor)
    ab = np.abs(to_np(a).astype(np.float64) - to_np(b).astype(np.float64))
    return {"max_rel": float(r.max(initial=0.0)), "rms_rel": float(np.sqrt(np.mean(r**2))) if r.size else 0.0,
            "max_abs": float(ab.max(initial=0.0)), "argmax_rel": int(r.argmax()) if r.size else -1}


def assert_close(actual, expected, rtol=1e-12, atol=0.0, name=""):
    a, e = to_np(actual).astype(np.float64), to_np(expected).astype(np.float64)
    assert a.shape == e.shape, f"{name}: shape {a.shape} != {e.shape}"
    bad = ~(np.abs(a - e) <= atol + rtol * np.abs(e))
    bad &= ~(np.isnan(a) & np.isnan(e))
    if bad.any():
        i = np.argwhere(bad)[0]
        s = err_stats(a, e)
        raise AssertionError(f"{name}: {bad.sum()}/{a.size} mismatches; first at {tuple(i)}: "
                             f"{a[tuple(i)]!r} vs {e[tuple(i)]!r}; max_rel={s['max_rel']:.3e}")
