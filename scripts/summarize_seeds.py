#!/usr/bin/env python3
"""Summarize three-seed metrics and optional paired model contrasts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
from scipy import stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", nargs="+", type=Path, required=True, help="pir-eval metric JSON files")
    parser.add_argument(
        "--reference",
        nargs="+",
        type=Path,
        help="paired reference files in the same seed order as --candidate",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--output", type=Path, default=Path("output/seed_summary.json"))
    return parser


def _metrics(path: Path) -> Dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and isinstance(payload.get("metrics"), Mapping):
        payload = payload["metrics"]
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} does not contain a metrics mapping")
    result = {}
    for key, value in payload.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if math.isfinite(numeric) and numeric >= 0.0:
                result[str(key)] = numeric
    if not result:
        raise ValueError(f"{path} contains no available numeric metrics")
    return result


def _summary(values: Sequence[float]) -> Dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    sample_sd = float(array.std(ddof=1)) if len(array) > 1 else None
    return {
        "n": len(values),
        "values": [float(value) for value in array],
        "mean": float(array.mean()),
        "sample_sd": sample_sd,
        "mean_percent": float(100.0 * array.mean()),
        "sample_sd_percentage_points": None if sample_sd is None else 100.0 * sample_sd,
    }


def _paired_summary(reference: Sequence[float], candidate: Sequence[float], confidence: float) -> Dict[str, object]:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    differences = cand - ref
    mean = float(differences.mean())
    sample_sd = float(differences.std(ddof=1)) if len(differences) > 1 else None
    if len(differences) > 1 and sample_sd is not None and sample_sd > 0.0:
        standard_error = sample_sd / math.sqrt(len(differences))
        radius = float(stats.t.ppf((1.0 + confidence) / 2.0, len(differences) - 1) * standard_error)
        interval = [mean - radius, mean + radius]
        p_value = float(stats.ttest_rel(cand, ref).pvalue)
    elif len(differences) > 1:
        interval = [mean, mean]
        p_value = 1.0 if mean == 0.0 else 0.0
    else:
        interval = None
        p_value = None
    return {
        "n": len(differences),
        "contrast": "candidate - reference",
        "differences": [float(value) for value in differences],
        "mean_difference": mean,
        "sample_sd": sample_sd,
        "confidence": confidence,
        "t_interval": interval,
        "paired_t_p_value": p_value,
        "mean_difference_percentage_points": 100.0 * mean,
        "t_interval_percentage_points": None if interval is None else [100.0 * value for value in interval],
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must lie strictly between zero and one")
    if args.reference is not None and len(args.reference) != len(args.candidate):
        raise ValueError("--reference and --candidate require the same number of paired files")

    candidate: List[Dict[str, float]] = [_metrics(path) for path in args.candidate]
    common = set.intersection(*(set(record) for record in candidate))
    result: Dict[str, object] = {
        "candidate_files": [str(path) for path in args.candidate],
        "candidate": {key: _summary([record[key] for record in candidate]) for key in sorted(common)},
    }
    if args.reference is not None:
        reference: List[Dict[str, float]] = [_metrics(path) for path in args.reference]
        paired_keys = common & set.intersection(*(set(record) for record in reference))
        result["reference_files"] = [str(path) for path in args.reference]
        result["reference"] = {key: _summary([record[key] for record in reference]) for key in sorted(paired_keys)}
        result["paired_contrast"] = {
            key: _paired_summary(
                [record[key] for record in reference],
                [record[key] for record in candidate],
                args.confidence,
            )
            for key in sorted(paired_keys)
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
