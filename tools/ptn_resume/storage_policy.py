"""Big-data storage placement policy for ReDimNet2 training and evaluation.

User rule (2026-11-09): big data written by these tools defaults under
``/root/group-shared/voiceprint`` and never under another ``/root`` path.
``/root`` itself keeps only code repositories and temporary logs.

* training products (run dir, checkpoints, training logs, exported weights)
  -> ``/root/group-shared/voiceprint/experiments/redimnet2/<实验名>/``
* evaluation products (embeddings, CSV, ledger)
  -> ``/root/group-shared/voiceprint/artifacts/redimnet2/eval-<日期>/``

This module is the single written-down copy of those two roots.  Shell
scripts cannot import it, so they carry the same literals and point here in a
comment; ``tests/test_storage_policy.py`` keeps the two in sync and refuses a
new default under ``/root/experiments`` or ``/root/codex_artifacts`` unless the
line carries ``# storage-policy-allow: <reason>``.

Historical artifacts that already exist under ``/root/experiments`` stay
referenced where they are: they are read-only inputs (for example model_115.pt
of the completed third PTN stage), not big-data defaults.  Those lines carry
the same allow marker and are listed in the review report.
"""

from __future__ import annotations

import datetime as dt
import re

GROUP_ROOT = "/root/group-shared/voiceprint"
EXPERIMENT_ROOT = GROUP_ROOT + "/experiments/redimnet2"
ARTIFACT_ROOT = GROUP_ROOT + "/artifacts/redimnet2"

# Prefixes that must never carry a *default* big-data write again.
FORBIDDEN_PREFIXES = ("/root/experiments", "/root/codex_artifacts")
# Whitelist marker for the lines that legitimately still mention them.
ALLOW_MARKER = "# storage-policy-allow:"

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _checked_name(name):
    if not isinstance(name, str) or _NAME.fullmatch(name) is None:
        raise ValueError("experiment/run name must be one path segment: %r" % (name,))
    return name


def experiment_dir(name):
    """Training run directory for one experiment name (no trailing slash)."""
    return EXPERIMENT_ROOT + "/" + _checked_name(name)


def eval_dir(date=None, run_name=None):
    """Evaluation artifact directory; date defaults to today in UTC (YYYYMMDD)."""
    if date is None:
        date = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")
    if not isinstance(date, str) or re.fullmatch(r"[0-9]{8}", date) is None:
        raise ValueError("eval date must be YYYYMMDD: %r" % (date,))
    path = ARTIFACT_ROOT + "/eval-" + date
    return path if run_name is None else path + "/" + _checked_name(run_name)
