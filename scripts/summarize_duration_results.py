#!/usr/bin/env python3
"""Recompute CPU statistics/plots from saved duration evaluation scores."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from duration_ft.metrics import score_metrics, paired_speaker_bootstrap
from duration_ft.plots import plot_duration_heatmap, plot_relative_full, plot_score_distributions


def _tag(x):
    return "full" if str(x) == "full" else format(float(x), "g")


def _cell_identity(c):
    cohort = str(c.get("cohort_id", c.get("matrix", "")))
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", cohort) or cohort in (".", ".."):
        raise ValueError("cohort must be an explicit safe name")
    return (cohort, _tag(c["enroll_duration"]),
            _tag(c.get("test_duration", c.get("verify_duration"))))


def _read_scores(summary_path, cell):
    cohort, left, right = _cell_identity(cell)
    supplied = cell.get("scores_csv") or cell.get("scores_path")
    p = Path(supplied) if supplied else Path(cohort) / f"{left}x{right}" / "scores.csv"
    if not p.is_absolute():
        p = summary_path.parent / p
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    if cell.get("scores_sha256") and cell["scores_sha256"] != digest:
        raise ValueError("score file SHA does not match evaluation summary")
    with p.open(newline="") as f:
        rows = list(csv.DictReader(f))
    ids = [r["trial_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate trial IDs: {p}")
    if not rows:
        raise ValueError(f"empty score table: {p}")
    ok = [r for r in rows if r["status"] == "ok"]
    # No silently dropping scores marked successful but NaN, empty or malformed.
    y, scores = [int(r["label"]) for r in ok], [float(r["score"]) for r in ok]
    if any(label not in (0, 1) for label in y) or any(not math.isfinite(s) for s in scores):
        raise ValueError("successful rows must have finite scores and binary labels")
    if int(cell["expected_trials"]) <= 0:
        raise ValueError("expected_trials must be positive")
    if len(rows) != int(cell["expected_trials"]):
        raise ValueError("score table must preserve every expected trial, including failures")
    complete = len(ok) == len(rows) == int(cell["expected_trials"])
    metrics = score_metrics(y, scores) if complete and set(y) == {0, 1} else None
    return {"path": str(p), "sha256": digest,
            "rows": rows, "ok": ok, "metrics": metrics, "complete": complete,
            "failed": len(rows) - len(ok), "coverage": len(ok) / int(cell["expected_trials"]),
            "metric_reason": None if metrics is not None else "incomplete fixed trial cohort or missing target class"}


def summarize(summary_path, output_dir, *, compare_summary=None, n_bootstrap=500, seed=20261023, make_plots=True):
    summary_path, output_dir = Path(summary_path).resolve(), Path(output_dir).resolve()
    source = json.loads(summary_path.read_text())
    other_path = Path(compare_summary).resolve() if compare_summary else None
    other = json.loads(other_path.read_text()) if other_path else None
    counterparts = {_cell_identity(c): c for c in other["cells"]} if other else {}
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"source_summary": str(summary_path), "source_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
              "comparison_summary": str(other_path) if other_path else None,
              "decision_rule": "score>=threshold", "cells": [], "plots": [],
              "limitations": ["EER/minDCF describe test discrimination, not deployed thresholds.",
                              "Fixed thresholds must come from independent dev; never fit here.",
                              "Bootstrap is speaker-vertex paired B-A, conditional on trained models.",
                              "Partial cells retain coverage but are not painted as complete quality results."]}
    groups = {}
    for c in source["cells"]:
        cohort, left, right = key = _cell_identity(c)
        x = _read_scores(summary_path, c)
        r = {"cohort_id": cohort, "enroll_duration": left, "test_duration": right,
             "trialset_sha256": c["trialset_sha256"], "expected_trials": c["expected_trials"],
             "scored_trials": len(x["ok"]), "failed_trials": x["failed"], "complete": x["complete"],
             "coverage": x["coverage"], "scores_csv": x["path"], "scores_sha256": x["sha256"],
             "metrics": x["metrics"], "metric_reason": x["metric_reason"],
             "fixed_threshold": c.get("fixed_threshold", {"status": "unavailable"})}
        if key in counterparts:
            cc = counterparts[key]
            z = _read_scores(other_path, cc)
            identity_a = c.get("comparison_identity_sha256", source.get("run_identity", {}).get("comparison_identity_sha256"))
            identity_b = cc.get("comparison_identity_sha256", other.get("run_identity", {}).get("comparison_identity_sha256"))
            if not (x["complete"] and z["complete"] and x["metrics"] is not None and z["metrics"] is not None
                    and c["trialset_sha256"] == cc["trialset_sha256"] and identity_a and identity_a == identity_b):
                r["paired_bootstrap"] = {"status": "unavailable", "reason": "incomplete cells, unequal trials, or absent/mismatched audio/crop/preprocess comparison identity"}
            else:
                by_id = {q["trial_id"]: q for q in z["ok"]}
                if set(by_id) != {q["trial_id"] for q in x["ok"]}:
                    raise ValueError("comparison trial IDs disagree despite matching trialset SHA")
                aligned = [by_id[q["trial_id"]] for q in x["ok"]]
                for a, b in zip(x["ok"], aligned):
                    if any(a[k] != b[k] for k in ["label", "enroll", "verify", "enroll_speaker", "test_speaker"]):
                        raise ValueError("comparison trial metadata disagree")
                fa = c.get("fixed_threshold", {}).get("far_0.01", {})
                fb = cc.get("fixed_threshold", {}).get("far_0.01", {})
                ta, tb = fa.get("threshold"), fb.get("threshold")
                if ta is None or tb is None:
                    ta = tb = None
                r["paired_bootstrap"] = paired_speaker_bootstrap(
                    [int(q["label"]) for q in x["ok"]], [float(q["score"]) for q in x["ok"]],
                    [float(q["score"]) for q in aligned], [q["enroll_speaker"] for q in x["ok"]],
                    [q["test_speaker"] for q in x["ok"]], n_bootstrap=n_bootstrap, seed=seed,
                    threshold_a=ta, threshold_b=tb)
                r["paired_bootstrap"]["fixed_threshold_source"] = "far_0.01 independent dev" if ta is not None else "unavailable"
        result["cells"].append(r)
        quality_ok = x["complete"] and r["metrics"] is not None
        flat = {**r, "eer_percent": r["metrics"]["eer_percent"] if quality_ok else None,
                "min_dcf_0.01": r["metrics"]["min_dcf"]["0.01"]["value"] if quality_ok else None,
                "min_dcf_0.001": r["metrics"]["min_dcf"]["0.001"]["value"] if quality_ok else None}
        for target in ["0.01", "0.001"]:
            f = r["fixed_threshold"].get("far_" + target, {})
            for metric in ["far_percent", "frr_percent"]:
                flat[metric + "_at_" + target] = f.get(metric) if quality_ok else None
        groups.setdefault(cohort, []).append(flat)
        if make_plots and quality_ok and left == right == "full":
            result["plots"].append(plot_score_distributions(
                [int(q["label"]) for q in x["ok"]], [float(q["score"]) for q in x["ok"]],
                output_dir / cohort / "full-score-distribution", title=f"Full/full scores: {cohort}"))
    for cohort, cells in groups.items():
        if make_plots:
            result["plots"].append(plot_duration_heatmap(cells, "eer_percent", output_dir / cohort / "eer", title="EER", unit="%"))
            result["plots"].append(plot_duration_heatmap(cells, "coverage", output_dir / cohort / "coverage", title="Scored trial coverage", unit="fraction"))
            for p in ["0.01", "0.001"]:
                result["plots"].append(plot_duration_heatmap(cells, "min_dcf_" + p, output_dir / cohort / ("minDCF-" + p), title="Normalized minDCF, Ptarget=" + p, unit="normalized cost"))
                for metric in ["far_percent", "frr_percent"]:
                    result["plots"].append(plot_duration_heatmap(cells, metric + "_at_" + p,
                        output_dir / cohort / (metric + "-devFAR-" + p),
                        title=metric + ", independent dev FAR=" + p, unit="%"))
            if all(c["complete"] for c in cells):
                result["plots"].append(plot_relative_full(cells, "eer_percent", output_dir / cohort / "eer-minus-full"))
            else:
                result["plots"].append({"status": "unavailable", "reason": "relative-full requires complete identical trial cohorts"})
    (output_dir / "statistics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--compare-summary")
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--seed", type=int, default=20261023)
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args()
    r = summarize(a.summary, a.output_dir, compare_summary=a.compare_summary,
                  n_bootstrap=a.bootstrap, seed=a.seed, make_plots=not a.no_plots)
    print(json.dumps({"cells": len(r["cells"]), "plots_generated": sum(x["status"] == "ok" for x in r["plots"]),
                      "output": str(Path(a.output_dir) / "statistics.json")}))


if __name__ == "__main__":
    main()
