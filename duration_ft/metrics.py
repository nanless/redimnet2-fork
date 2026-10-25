"""CPU verification metrics. Higher scores mean same speaker; accept score >= t.

EER/minDCF thresholds describe test discrimination, never deployment calibration.
No GPU, model, filesystem or network operations occur in this module.
"""
from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np


def _inputs(labels, scores, weights=None):
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s) or not len(y):
        raise ValueError("labels/scores must be nonempty equal-length vectors")
    if not np.all((y == 0) | (y == 1)) or not np.all(np.isfinite(s)):
        raise ValueError("labels must be binary and every score finite")
    y = y.astype(bool)
    w = np.ones(len(y), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != s.shape or not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite nonnegative and match scores")
    wt, wn = float(w[y].sum()), float(w[~y].sum())
    if not (0 < wt < math.inf and 0 < wn < math.inf and math.isfinite(wt + wn)):
        raise ValueError("positive finite weight required in both target classes")
    return y, s, w


def _prepared(y, s):
    order = np.argsort(-s, kind="stable")
    ss = s[order]
    ends = np.r_[np.flatnonzero(ss[:-1] != ss[1:]), len(ss) - 1]
    return order, ends, ss[ends]


def _roc(y, w, prepared):
    order, ends, group_scores = prepared
    ys, ws = y[order], w[order]
    wt, wn = float(w[y].sum()), float(w[~y].sum())
    far = np.r_[0.0, np.cumsum(ws * (~ys))[ends] / wn]
    frr = np.r_[1.0, 1.0 - np.cumsum(ws * ys)[ends] / wt]
    # The first row rejects everything, even if max(score) is float64.max.
    return np.clip(far, 0, 1), np.clip(frr, 0, 1), np.r_[np.inf, group_scores]


def _threshold_fields(t):
    if math.isinf(float(t)):
        return {"threshold": None, "threshold_kind": "reject_all" if t > 0 else "accept_all"}
    return {"threshold": float(t), "threshold_kind": "score_greater_equal"}


def _metrics_core(y, s, w, prepared, p_targets):
    far, frr, ts = _roc(y, w, prepared)
    difference = far - frr
    i = int(np.flatnonzero(difference >= 0)[0])
    exact = abs(float(difference[i])) <= 1e-15
    if exact:
        eer, eer_t = float(far[i]), float(ts[i])
    else:
        lo = i - 1
        alpha = float(-difference[lo] / (difference[i] - difference[lo]))
        eer = float(far[lo] + alpha * (far[i] - far[lo]))
        eer_t = None
    dcfs = {}
    for p in p_targets:
        p = float(p)
        if not math.isfinite(p) or not 0 < p < 1:
            raise ValueError("Ptarget must be strictly between zero and one")
        costs = (p * frr + (1 - p) * far) / min(p, 1 - p)
        k = int(np.argmin(costs))
        dcfs[format(p, ".12g")] = {"value": float(costs[k]), **_threshold_fields(ts[k]),
                                  "p_target": p, "c_miss": 1, "c_fa": 1}
    if not dcfs:
        raise ValueError("at least one Ptarget is required")
    tpr = 1 - frr
    auc = float(np.sum(np.diff(far) * (tpr[1:] + tpr[:-1]) * .5))
    return {"eer": eer, "eer_percent": 100 * eer, "eer_threshold": eer_t,
            "eer_interpolated": not exact, "auc": auc,
            "min_dcf": dcfs, "n_trials": len(y), "n_target": int(y.sum()),
            "n_nontarget": int((~y).sum()), "target_weight": float(w[y].sum()),
            "nontarget_weight": float(w[~y].sum()), "decision_rule": "score>=threshold",
            "threshold_use": "EER/minDCF thresholds are descriptive only; use independent dev calibration"}


def score_metrics(labels, scores, *, weights=None, p_targets=(0.01, 0.001)):
    """Return strict-JSON-safe EER (fraction and %), AUC, normalized minDCF.

    Ties are accepted/rejected together. Interpolated EER has no deployable
    threshold. A minDCF reject-all endpoint has threshold=None and an explicit
    threshold_kind. Weights change empirical rates, never score ordering.
    """
    y, s, w = _inputs(labels, scores, weights)
    return _metrics_core(y, s, w, _prepared(y, s), tuple(p_targets))


def fixed_threshold_metrics(labels, scores, threshold, *, weights=None):
    """Evaluate a supplied threshold without fitting it on these labels."""
    y, s, w = _inputs(labels, scores, weights)
    t = float(threshold)
    if math.isnan(t):
        raise ValueError("threshold cannot be NaN")
    accept = s >= t
    tp, fn = float(w[y & accept].sum()), float(w[y & ~accept].sum())
    fp, tn = float(w[~y & accept].sum()), float(w[~y & ~accept].sum())
    far, frr = fp / (fp + tn), fn / (tp + fn)
    return {**_threshold_fields(t), "decision_rule": "score>=threshold", "far": far,
            "frr": frr, "far_percent": far * 100, "frr_percent": frr * 100,
            "tpr": 1 - frr, "tnr": 1 - far, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "n_trials": len(y), "n_target": int(y.sum()), "n_nontarget": int((~y).sum()),
            "weighted_counts": weights is not None}


def calibrate_far_threshold(labels, scores, *, target_far=0.01):
    """Conservative empirical dev FAR threshold, with score ties kept whole.

    Caller must establish dev/test speaker independence. No population FAR
    guarantee is implied by an empirical rate or by zero observed errors.
    """
    alpha = float(target_far)
    if not math.isfinite(alpha) or not 0 <= alpha < 1:
        raise ValueError("target_far must be in [0,1)")
    y, s, w = _inputs(labels, scores)
    far, frr, ts = _roc(y, w, _prepared(y, s))
    k = int(np.flatnonzero(far <= alpha)[-1])
    t = float(ts[k])
    if math.isinf(t):
        with np.errstate(over="ignore"):
            t = float(np.nextafter(float(s.max()), np.inf))
        if not math.isfinite(t):
            raise ValueError("cannot represent a finite reject-all threshold above float64.max")
    return {"threshold": t, "target_far": alpha, "dev_far": float(far[k]),
            "dev_frr": float(frr[k]), "n_trials": len(y), "n_target": int(y.sum()),
            "n_nontarget": int((~y).sum()), "decision_rule": "score>=threshold",
            "scope": "independent dev only; caller must verify speaker/sample separation",
            "limitations": "empirical FAR only; correlated trials and finite data do not guarantee population FAR"}


def paired_speaker_bootstrap(labels, scores_a, scores_b, speaker_enroll, speaker_test, *,
                             n_bootstrap=500, seed=20261023, confidence=0.95,
                             threshold_a=None, threshold_b=None):
    """Speaker vertex resampling; report B-minus-A paired differences.

    Target weight m_s; non-target weight m_s*m_t. Both speaker endpoints and
    every occurrence/crop remain linked. This explicit design is not claimed
    to reproduce a particular NIST implementation. CI is conditional on fixed
    supplied dev thresholds and trained models, not training-seed uncertainty.
    """
    y, a, ones = _inputs(labels, scores_a)
    _, b, _ = _inputs(labels, scores_b)
    ea, eb = list(speaker_enroll), list(speaker_test)
    if len(ea) != len(y) or len(eb) != len(y) or any(not isinstance(x, str) or not x for x in ea + eb):
        raise ValueError("nonempty string speaker IDs required for every endpoint")
    if any(bool(label) != (x == z) for label, x, z in zip(y, ea, eb)):
        raise ValueError("speaker identity equality must agree with target label")
    if not isinstance(n_bootstrap, int) or n_bootstrap < 20 or not 0 < confidence < 1:
        raise ValueError("use at least20 bootstrap attempts and confidence in (0,1)")
    if (threshold_a is None) != (threshold_b is None):
        raise ValueError("supply both fixed dev thresholds or neither")
    speakers = sorted(set(ea + eb))
    if len(speakers) < 2:
        raise ValueError("at least two speakers required")
    lookup = {x: i for i, x in enumerate(speakers)}
    ia, ib = np.array([lookup[x] for x in ea]), np.array([lookup[x] for x in eb])
    pa, pb = _prepared(y, a), _prepared(y, b)

    def measure(s, weights, prepared, threshold):
        r = _metrics_core(y, s, weights, prepared, (0.01, 0.001))
        out = {"eer": r["eer"], "eer_percent": r["eer_percent"],
               **{"min_dcf_" + k: v["value"] for k, v in r["min_dcf"].items()}}
        if threshold is not None:
            f = fixed_threshold_metrics(y, s, threshold, weights=weights)
            out.update(far=f["far"], frr=f["frr"])
        return out

    base_a = measure(a, ones, pa, threshold_a)
    base_b = measure(b, ones, pb, threshold_b)
    draws = {k: [] for k in base_a}
    marginal_draws_a = {k: [] for k in base_a}
    marginal_draws_b = {k: [] for k in base_b}
    rng, invalid = random.Random(seed), 0
    for _ in range(n_bootstrap):
        counts = Counter(rng.choices(range(len(speakers)), k=len(speakers)))
        m = np.array([counts[i] for i in range(len(speakers))], dtype=float)
        weights = np.where(y, m[ia], m[ia] * m[ib])
        if weights[y].sum() <= 0 or weights[~y].sum() <= 0:
            invalid += 1
            continue
        ra, rb = measure(a, weights, pa, threshold_a), measure(b, weights, pb, threshold_b)
        for k in draws:
            draws[k].append(rb[k] - ra[k])
            marginal_draws_a[k].append(ra[k])
            marginal_draws_b[k].append(rb[k])
    valid = n_bootstrap - invalid
    enough = valid >= max(10, math.ceil(0.8 * n_bootstrap))
    tail = (1 - confidence) / 2
    differences = {}
    for k, values in draws.items():
        interval = np.quantile(values, [tail, 1 - tail]).tolist() if enough else [None, None]
        differences[k] = {"estimate": base_b[k] - base_a[k], "lower": interval[0],
                          "upper": interval[1], "unit": "percentage_points" if k == "eer_percent" else "fraction_or_normalized_cost"}
    return {"status": "ok" if enough else "insufficient_valid_resamples", "method": "paired speaker vertex weights",
            "target_weight": "m_s", "nontarget_weight": "m_s*m_t", "difference_direction": "B-A",
            "speaker_count": len(speakers), "attempts": n_bootstrap, "valid": valid,
            "invalid_missing_class": invalid, "seed": seed, "confidence": confidence,
            "differences": differences, "point_a": base_a, "point_b": base_b,
            "marginal_a": _marginal_intervals(base_a, marginal_draws_a, enough, confidence),
            "marginal_b": _marginal_intervals(base_b, marginal_draws_b, enough, confidence),
            "limitations": "speaker-correlated conditional CI; not IID-trial, not training-seed or dev-threshold uncertainty"}


def _marginal_intervals(point, draws, enough, confidence):
    tail = (1 - confidence) / 2
    intervals = {}
    for name, estimate in point.items():
        interval = np.quantile(draws[name], [tail, 1 - tail]).tolist() if enough else [None, None]
        unit = "percent" if name == "eer_percent" else ("normalized_cost" if name.startswith("min_dcf_") else "fraction")
        intervals[name] = {"estimate": estimate, "lower": interval[0], "upper": interval[1], "unit": unit}
    return intervals


def single_speaker_bootstrap(labels, scores, speaker_enroll, speaker_test, *,
                             n_bootstrap=500, seed=20261023, confidence=0.95,
                             threshold=None):
    """Absolute EER/minDCF percentile CIs from speaker vertex resampling.

    The actual speaker universe is inferred from both endpoints (40 for the
    complete Vox1-O protocol), never silently truncated or hard-coded. Target
    weight is m_s, non-target weight m_s*m_t, using exactly the same draw order
    as paired_speaker_bootstrap. A single model is measured once per draw.
    Optional FAR/FRR CIs condition on an already frozen independent-dev
    threshold. These are percentile intervals, need not be symmetric around
    or contain the empirical point estimate in a small sample.
    """
    y, s, ones = _inputs(labels, scores)
    ea, eb = list(speaker_enroll), list(speaker_test)
    if len(ea) != len(y) or len(eb) != len(y) or any(not isinstance(x, str) or not x for x in ea + eb):
        raise ValueError("nonempty string speaker IDs required for every endpoint")
    if any(bool(label) != (x == z) for label, x, z in zip(y, ea, eb)):
        raise ValueError("speaker identity equality must agree with target label")
    if not isinstance(n_bootstrap, int) or n_bootstrap < 20 or not 0 < confidence < 1:
        raise ValueError("use at least20 bootstrap attempts and confidence in (0,1)")
    speakers = sorted(set(ea + eb))
    if len(speakers) < 2:
        raise ValueError("at least two speakers required")
    lookup = {x: i for i, x in enumerate(speakers)}
    ia, ib = np.array([lookup[x] for x in ea]), np.array([lookup[x] for x in eb])
    prepared = _prepared(y, s)

    def measure(weights):
        r = _metrics_core(y, s, weights, prepared, (0.01, 0.001))
        out = {"eer": r["eer"], "eer_percent": r["eer_percent"],
               **{"min_dcf_" + k: v["value"] for k, v in r["min_dcf"].items()}}
        if threshold is not None:
            f = fixed_threshold_metrics(y, s, threshold, weights=weights)
            out.update(far=f["far"], frr=f["frr"])
        return out

    point = measure(ones)
    draws = {k: [] for k in point}
    rng, invalid = random.Random(seed), 0
    for _ in range(n_bootstrap):
        counts = Counter(rng.choices(range(len(speakers)), k=len(speakers)))
        m = np.array([counts[i] for i in range(len(speakers))], dtype=float)
        weights = np.where(y, m[ia], m[ia] * m[ib])
        if weights[y].sum() <= 0 or weights[~y].sum() <= 0:
            invalid += 1
            continue
        result = measure(weights)
        for k in draws:
            draws[k].append(result[k])
    valid = n_bootstrap - invalid
    enough = valid >= max(10, math.ceil(0.8 * n_bootstrap))
    return {"status": "ok" if enough else "insufficient_valid_resamples", "method": "speaker vertex weights",
            "target_weight": "m_s", "nontarget_weight": "m_s*m_t", "interval_method": "percentile",
            "speaker_count": len(speakers), "attempts": n_bootstrap, "valid": valid,
            "invalid_missing_class": invalid, "seed": seed, "confidence": confidence,
            "point": point, "intervals": _marginal_intervals(point, draws, enough, confidence),
            "limitations": "speaker-correlated conditional CI; not IID-trial, not training-seed or dev-threshold uncertainty; small-sample percentile intervals can be asymmetric and exclude the point estimate"}
