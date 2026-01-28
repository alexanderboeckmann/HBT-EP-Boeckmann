#!/usr/bin/env python3
"""
Compare base phase vs sin/cos wrapped phase using a wrap-safe "GA-style percent" score.

Why this exists
---------------
The genetic algorithm (GA) uses a percent score that is effectively a full-scale-normalized MAE:

  score = 100 * mean(|y - y_hat| / max(|y|))

For phase, classic MAPE is problematic at the +/-pi discontinuity. This script compares:
- base phase prediction (mp*) directly in angle space, with wrap-safe error
- sin/cos predictions (mps* + mpc*) reconstructed via atan2, then compared in angle space

The reported "GA-style phase percent" is:

  100 * mean(|wrapToPi(theta_hat - theta_true)| / pi)

Inputs
------
Point each argument at a directory containing:
- results_*_<data_type>_true.npy
- results_*_<data_type>_pred.npy
- normalization_*.npz (optional but recommended; used to de-normalize)

These are produced by the analysis classes and by the optimization scripts (per-individual folders).

Example
-------
python scripts/evaluation/compare_phase_wrapped.py \
  --mp_dir  data/optimization_results/run_parallel_mp2_.../individual_... \
  --mps_dir data/optimization_results/run_parallel_mps2_.../individual_... \
  --mpc_dir data/optimization_results/run_parallel_mpc2_.../individual_... \
  --mode 2
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _find_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern!r} in {directory}")
    if len(matches) > 1:
        # Prefer the shortest name deterministically, but tell the user what happened.
        # (This avoids ambiguous results dirs containing multiple notebook_types/states.)
        matches = sorted(matches, key=lambda p: (len(p.name), p.name))
    return matches[0]


def _load_true_pred(results_dir: Path, data_type: str) -> Tuple[np.ndarray, np.ndarray]:
    true_path = _find_one(results_dir, f"results_*_{data_type}_true.npy")
    pred_path = _find_one(results_dir, f"results_*_{data_type}_pred.npy")
    true = np.load(true_path)
    pred = np.load(pred_path)
    return np.asarray(true).reshape(-1), np.asarray(pred).reshape(-1)


def _load_ma_norm(results_dir: Path) -> float:
    # In this repo, normalization is saved as:
    #   normalization_<notebook_type>_state_<state>.npz
    # and includes: ma_norm
    norm_files = sorted(results_dir.glob("normalization_*_state_*.npz"))
    if not norm_files:
        return 1.0
    norm_path = sorted(norm_files, key=lambda p: (len(p.name), p.name))[0]
    data = np.load(norm_path)
    ma_norm = float(data.get("ma_norm", 1.0))
    if not np.isfinite(ma_norm) or ma_norm == 0.0:
        return 1.0
    return ma_norm


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    # Wrap to [-pi, pi] using atan2(sin, cos) (stable, vectorized).
    return np.arctan2(np.sin(angle), np.cos(angle))


def _phase_percent_score(theta_true: np.ndarray, theta_pred: np.ndarray) -> float:
    # "GA-style phase percent": average absolute wrapped error, normalized by pi.
    delta = _wrap_to_pi(theta_pred - theta_true)
    return float(np.mean(np.abs(delta)) / math.pi * 100.0)


def _summary(theta_true: np.ndarray, theta_pred: np.ndarray) -> str:
    delta = _wrap_to_pi(theta_pred - theta_true)
    mae_rad = float(np.mean(np.abs(delta)))
    rmse_rad = float(np.sqrt(np.mean(delta**2)))
    mae_deg = mae_rad * 180.0 / math.pi
    rmse_deg = rmse_rad * 180.0 / math.pi
    pct = _phase_percent_score(theta_true, theta_pred)
    return (
        f"GA-style phase %: {pct:.3f}% | "
        f"MAAE: {mae_rad:.4f} rad ({mae_deg:.2f} deg) | "
        f"RMSE: {rmse_rad:.4f} rad ({rmse_deg:.2f} deg)"
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare base phase vs sin/cos wrapped phase.")
    parser.add_argument("--mp_dir", type=str, required=True, help="Directory containing results for mp<mode>.")
    parser.add_argument("--mps_dir", type=str, required=True, help="Directory containing results for mps<mode> (sin).")
    parser.add_argument("--mpc_dir", type=str, required=True, help="Directory containing results for mpc<mode> (cos).")
    parser.add_argument("--mode", type=int, default=2, help="Mode index (1-4). Default: 2")
    args = parser.parse_args(argv)

    if args.mode not in (1, 2, 3, 4):
        raise SystemExit("--mode must be 1, 2, 3, or 4")

    mp_dir = Path(args.mp_dir).expanduser().resolve()
    mps_dir = Path(args.mps_dir).expanduser().resolve()
    mpc_dir = Path(args.mpc_dir).expanduser().resolve()

    mp = f"mp{args.mode}"
    mps = f"mps{args.mode}"
    mpc = f"mpc{args.mode}"

    # Load normalized arrays.
    mp_true_n, mp_pred_n = _load_true_pred(mp_dir, mp)
    mps_true_n, mps_pred_n = _load_true_pred(mps_dir, mps)
    mpc_true_n, mpc_pred_n = _load_true_pred(mpc_dir, mpc)

    # De-normalize (critical: sin/cos need to be in [-1, 1] units for atan2 to make sense).
    mp_norm = _load_ma_norm(mp_dir)
    mps_norm = _load_ma_norm(mps_dir)
    mpc_norm = _load_ma_norm(mpc_dir)

    theta_true = mp_true_n * mp_norm
    theta_pred_base = mp_pred_n * mp_norm

    sin_true = mps_true_n * mps_norm
    sin_pred = mps_pred_n * mps_norm
    cos_true = mpc_true_n * mpc_norm
    cos_pred = mpc_pred_n * mpc_norm

    # Align lengths conservatively.
    n = min(len(theta_true), len(theta_pred_base), len(sin_pred), len(cos_pred), len(sin_true), len(cos_true))
    if n == 0:
        raise SystemExit("No overlapping samples to compare (arrays are empty).")

    theta_true = theta_true[:n]
    theta_pred_base = theta_pred_base[:n]
    sin_true = sin_true[:n]
    cos_true = cos_true[:n]
    sin_pred = sin_pred[:n]
    cos_pred = cos_pred[:n]

    # Reconstruct phase from sin/cos.
    theta_true_from_sc = np.arctan2(sin_true, cos_true)
    theta_pred_from_sc = np.arctan2(sin_pred, cos_pred)

    print(f"Comparing mode {args.mode} using n={n} samples")
    print(f"mp_dir : {mp_dir}")
    print(f"mps_dir: {mps_dir}")
    print(f"mpc_dir: {mpc_dir}")
    print("")
    print("Base phase (mp*) vs mp* true:")
    print("  " + _summary(theta_true, theta_pred_base))
    print("")
    print("Reconstructed phase (atan2(mps, mpc)) vs mp* true:")
    print("  " + _summary(theta_true, theta_pred_from_sc))
    print("")
    # Sanity check: sin/cos ground truth reconstructed should match mp truth up to wrapping.
    sanity = _phase_percent_score(theta_true, theta_true_from_sc)
    print("Sanity (reconstructed TRUE from mps/mpc vs mp* true):")
    print(f"  GA-style phase %: {sanity:.3f}% (should be near 0 if files correspond)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

