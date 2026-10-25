"""Standalone scientific PNG/PDF plots from saved observations only."""
from __future__ import annotations

from pathlib import Path
import math

import numpy as np


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["Noto Sans CJK SC", "Arial Unicode MS", "PingFang SC", "WenQuanYi Zen Hei"]:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    return plt


def _save(fig, output_prefix):
    p = Path(output_prefix)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Decimal duration/prior names are part of the prefix, not extensions.
    paths = {"png": str(p) + ".png", "pdf": str(p) + ".pdf"}
    fig.savefig(paths["png"], dpi=180, bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    return paths


def _duration_order(x):
    if str(x) == "full":
        return (1, 0)
    v = float(x)
    if not math.isfinite(v) or v <= 0:
        raise ValueError("duration must be positive finite or full")
    return (0, v)


def _cohort(records, *, require_trialset=False):
    cohorts = {str(x.get("cohort_id", x.get("matrix", ""))) for x in records}
    if len(cohorts) != 1 or "" in cohorts:
        raise ValueError("plot only one explicitly identified trial cohort")
    trialsets = {x.get("trialset_sha256") for x in records}
    if require_trialset and (len(trialsets) != 1 or None in trialsets or "" in trialsets):
        raise ValueError("relative-to-full plot requires identical explicit trialset SHA")
    return next(iter(cohorts))


def plot_duration_heatmap(records, value_key, output_prefix, *, title=None, unit="%", diverging=False):
    """Records have enroll_duration/test_duration, cohort_id and flat value_key.

    Missing values remain gray, never zero. Duplicate cells are rejected to
    avoid silently mixing models, trials or repeated measurements.
    """
    records = list(records)
    if not records:
        raise ValueError("no measured records to plot")
    cohort = _cohort(records)
    left = sorted({str(x["enroll_duration"]) for x in records}, key=_duration_order)
    right = sorted({str(x["test_duration"]) for x in records}, key=_duration_order)
    data = np.full((len(left), len(right)), np.nan)
    seen = set()
    for r in records:
        key = (str(r["enroll_duration"]), str(r["test_duration"]))
        if key in seen:
            raise ValueError("duplicate duration cell")
        seen.add(key)
        value = r.get(value_key)
        if value is not None:
            if not math.isfinite(float(value)):
                raise ValueError("nonfinite measured plot value")
            data[left.index(key[0]), right.index(key[1])] = float(value)
    if not np.any(np.isfinite(data)):
        return {"status": "unavailable", "reason": "no finite measurements", "files": {}}
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(max(5, len(right) * .85), max(4.2, len(left) * .7)))
    cmap = plt.get_cmap("RdBu_r" if diverging else "viridis_r").copy()
    cmap.set_bad("#e5e7eb")
    kwargs = {}
    if diverging:
        bound = max(float(np.nanmax(np.abs(data))), 1e-12)
        kwargs = {"vmin": -bound, "vmax": bound}
    im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, **kwargs)
    for i in range(len(left)):
        for k in range(len(right)):
            v = data[i, k]
            ax.text(k, i, "missing" if np.isnan(v) else f"{v:.3f}", ha="center", va="center",
                    fontsize=8, bbox={"facecolor": "white", "alpha": .72, "edgecolor": "none", "pad": 1})
    ax.set_xticks(range(len(right)), right)
    ax.set_yticks(range(len(left)), left)
    ax.set_xlabel("Test duration (seconds; full = natural length)")
    ax.set_ylabel("Enrollment duration (seconds)")
    ax.set_title((title or value_key) + "\n" + cohort)
    fig.colorbar(im, ax=ax, label=unit, shrink=.85)
    fig.tight_layout()
    paths = _save(fig, output_prefix)
    plt.close(fig)
    return {"status": "ok", "files": paths, "cohort_id": cohort,
            "measured_cells": int(np.isfinite(data).sum()), "missing_cells": int(np.isnan(data).sum())}


def plot_score_distributions(labels, scores, output_prefix, *, title="Cosine score distributions", bins=60):
    from .metrics import _inputs
    y, s, _ = _inputs(labels, scores)
    if not isinstance(bins, int) or bins < 2:
        raise ValueError("at least two histogram bins required")
    lo, hi = float(s.min()), float(s.max())
    if lo == hi:
        lo, hi = lo - .05, hi + .05
    edges = np.linspace(lo, hi, bins + 1)
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.hist(s[~y], bins=edges, alpha=.55, label=f"Different speaker (n={(~y).sum()})", color="#c2410c")
    ax.hist(s[y], bins=edges, alpha=.55, label=f"Same speaker (n={y.sum()})", color="#0369a1")
    ax.set(xlabel="Score (higher means same speaker)", ylabel="Trial count", title=title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    paths = _save(fig, output_prefix)
    plt.close(fig)
    return {"status": "ok", "files": paths, "n_trials": len(y)}


def plot_relative_full(records, value_key, output_prefix, *, title=None, unit="percentage points"):
    records = [dict(x) for x in records]
    _cohort(records, require_trialset=True)
    if any(x.get("complete") is not True for x in records):
        raise ValueError("relative-to-full requires complete cells, not differing scored subsets")
    base = [x for x in records if str(x["enroll_duration"]) == "full" and str(x["test_duration"]) == "full"]
    if len(base) != 1 or base[0].get(value_key) is None:
        return {"status": "unavailable", "reason": "same-cohort full/full measurement missing", "files": {}}
    baseline = float(base[0][value_key])
    for r in records:
        r["change_from_full"] = None if r.get(value_key) is None else float(r[value_key]) - baseline
    result = plot_duration_heatmap(records, "change_from_full", output_prefix,
                                   title=title or f"{value_key} minus full/full", unit=unit, diverging=True)
    result["baseline"] = baseline
    result["difference_direction"] = "cell minus full/full, same trial cohort"
    return result
