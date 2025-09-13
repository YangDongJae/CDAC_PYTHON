# core/registration/stats.py
from __future__ import annotations
import numpy as np

try:
    import torch
except ImportError:
    torch = None

def complex_stats(x):
    """|z| 기준 min/max, NaN 포함 여부"""
    if torch is not None and torch.is_tensor(x):
        mag = torch.abs(x)
        m = mag[~torch.isnan(mag)] if mag.is_floating_point() else mag.reshape(-1)
        if m.numel() == 0:
            return float("nan"), float("nan"), True
        return float(m.min().item()), float(m.max().item()), bool(torch.isnan(x.real).any() or torch.isnan(x.imag).any())
    a = np.asarray(x)
    mag = np.abs(a)
    m = mag[~np.isnan(mag)]
    if m.size == 0:
        return float("nan"), float("nan"), True
    any_nan = bool(np.isnan(a.real).any() if np.iscomplexobj(a) else np.isnan(a).any())
    return float(np.min(m)), float(np.max(m)), any_nan

def summarize_minmax(layers_data):
    for k in ("ILM", "NFL", "ISOS", "RPE"):
        a = layers_data[k]
        if torch is not None and torch.is_tensor(a):
            a = a.detach().cpu().numpy()
        print(f"{k}: min={np.nanmin(a):.4f}, max={np.nanmax(a):.4f}")

def report_layer_order(layers_data, depth_per_pixel_mm: float | None = None):
    import numpy as np
    ILM  = np.asarray(layers_data["ILM"])
    NFL  = np.asarray(layers_data["NFL"])
    ISOS = np.asarray(layers_data["ISOS"])
    RPE  = np.asarray(layers_data["RPE"])

    diffs = {
        "NFL-ILM":  NFL - ILM,
        "ISOS-NFL": ISOS - NFL,
        "RPE-ISOS": RPE  - ISOS,
    }
    total = ILM.size
    for name, d in diffs.items():
        d = d.astype(float)
        neg = d < 0
        worst = np.nanmin(d)
        pct = float(np.count_nonzero(neg)) / total * 100.0
        msg = f"{name}: min={np.nanmin(d):.3f}, mean={np.nanmean(d):.3f}, violations={np.count_nonzero(neg)} ({pct:.2f}%)"
        if depth_per_pixel_mm is not None and np.isfinite(worst):
            msg += f", worst_gap_mm={worst * depth_per_pixel_mm:.4f}"
        print(msg)