#!/usr/bin/env python3
"""
Plot phase-over-time comparisons for:
- true phase (mp*)
- base/raw phase prediction (mp* pred)
- sin/cos prediction reconstructed to phase via atan2(mps, mpc)

This is meant to mirror the kind of visual comparison used for amplitude, but for phase.

Inputs
------
Point each argument at a directory containing results saved by the analysis classes:
  - results_*_<data_type>_true.npy
  - results_*_<data_type>_pred.npy
  - results_*_<data_type>_time.npy
  - normalization_*.npz (optional but recommended; used to de-normalize)

Example
-------
python scripts/evaluation/plot_phase_comparison.py \
  --mp_dir  data/optimization_results/run_parallel_mp2_.../individual_.../mp2 \
  --mps_dir data/optimization_results/run_parallel_mp_sc2_.../individual_.../mps2 \
  --mpc_dir data/optimization_results/run_parallel_mp_sc2_.../individual_.../mpc2 \
  --mode 2 --unwrap --show
"""

from __future__ import annotations

import argparse
import math
import re
import json
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np


def _find_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern!r} in {directory}")
    if len(matches) > 1:
        # Prefer deterministic selection; avoids ambiguity in directories containing multiple result sets.
        matches = sorted(matches, key=lambda p: (len(p.name), p.name))
    return matches[0]


def _load_true_pred_time(results_dir: Path, data_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_path = _find_one(results_dir, f"results_*_{data_type}_true.npy")
    pred_path = _find_one(results_dir, f"results_*_{data_type}_pred.npy")
    time_path = _find_one(results_dir, f"results_*_{data_type}_time.npy")
    true = np.asarray(np.load(true_path)).reshape(-1)
    pred = np.asarray(np.load(pred_path)).reshape(-1)
    t = np.asarray(np.load(time_path)).reshape(-1)
    return true, pred, t


def _load_true_pred(results_dir: Path, data_type: str) -> Tuple[np.ndarray, np.ndarray]:
    true_path = _find_one(results_dir, f"results_*_{data_type}_true.npy")
    pred_path = _find_one(results_dir, f"results_*_{data_type}_pred.npy")
    true = np.asarray(np.load(true_path)).reshape(-1)
    pred = np.asarray(np.load(pred_path)).reshape(-1)
    return true, pred


def _load_ma_norm(results_dir: Path) -> float:
    """
    Load ma_norm from normalization_*.npz if present.

    Note:
    In this repo, the saved results arrays for mp/mps/mpc are already in physical units
    (true is raw, predictions are de-normalized before saving). So we DO NOT apply ma_norm
    when plotting/scoring. We keep this loader only for debugging / optional future use.
    """
    norm_files = sorted(results_dir.glob("normalization_*_state_*.npz"))
    if not norm_files:
        return 1.0
    norm_path = sorted(norm_files, key=lambda p: (len(p.name), p.name))[0]
    data = np.load(norm_path)
    ma_norm = float(data.get("ma_norm", 1.0))
    if not np.isfinite(ma_norm) or ma_norm == 0.0:
        return 1.0
    return ma_norm


def _repo_root_from_this_file() -> Path:
    # This file lives at <repo_root>/scripts/evaluation/plot_phase_comparison.py
    return Path(__file__).resolve().parents[2]


def _safe_mtime(p: Path) -> float:
    try:
        return float(p.stat().st_mtime)
    except Exception:
        return 0.0


def _list_run_dirs(results_root: Path) -> list[Path]:
    if not results_root.exists():
        return []
    return sorted([p for p in results_root.iterdir() if p.is_dir()], key=_safe_mtime, reverse=True)


def _match_latest_run(results_root: Path, mode: int, want: str) -> Path:
    """
    Find the most recent optimization run directory for:
    - want="mp":    run_*_mp<mode>_YYYY... or run_*_mp<mode>_...
    - want="mp_sc": run_*_(mp_sc|mpSC|phase_sc|phaseSC)<mode>_...
    """
    mode_s = str(mode)
    if want == "mp":
        pat = re.compile(rf"^run(_parallel)?_mp{mode_s}_.+")
    elif want == "mp_sc":
        pat = re.compile(rf"^run(_parallel)?_(mp_?sc|phase_?sc){mode_s}_.+")
    else:
        raise ValueError(f"Unknown want={want!r}")

    for run_dir in _list_run_dirs(results_root):
        if pat.match(run_dir.name):
            return run_dir
    raise FileNotFoundError(f"No matching run directory for {want} mode={mode} under {results_root}")


def _load_params(individual_dir: Path) -> dict[str, Any]:
    params_path = individual_dir / "parameters.json"
    if not params_path.exists():
        return {}
    try:
        return json.loads(params_path.read_text())
    except Exception:
        return {}


def _normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize the params schema across scripts.

    Optimization scripts write `parameters.json` with keys like:
      - state
      - selected_data_type
      - RESERVED_SHOT

    Analysis configs internally may use `reserved_shot`. We standardize to:
      - state (int)
      - reserved_shot (int)
    """
    out: dict[str, Any] = dict(params or {})
    if "reserved_shot" not in out and "RESERVED_SHOT" in out:
        out["reserved_shot"] = out.get("RESERVED_SHOT")
    # Make sure state/reserved_shot are ints when possible.
    for k in ("state", "reserved_shot"):
        if k in out:
            try:
                out[k] = int(out[k])
            except Exception:
                pass
    return out


def _infer_notebook_type(results_dir: Path) -> Optional[str]:
    """
    Infer notebook_type ('trimmed' or 'untrimmed') from result filenames in a directory.
    """
    if any(results_dir.glob("results_trimmed_state_*_*.npy")):
        return "trimmed"
    if any(results_dir.glob("results_untrimmed_state_*_*.npy")):
        return "untrimmed"
    return None

def _best_individual_from_csv(run_dir: Path) -> Optional[Path]:
    csv_path = run_dir / "hbt_optimization_results.csv"
    if not csv_path.exists():
        return None
    try:
        import pandas as pd  # optional dependency; repo already uses it in optimizers

        df = pd.read_csv(csv_path)
        if "mape" not in df.columns or "individual_id" not in df.columns:
            return None
        df = df[df["mape"].notna()]
        if df.empty:
            return None
        best_id = str(df.sort_values("mape").iloc[0]["individual_id"])
        ind = run_dir / f"individual_{best_id}"
        return ind if ind.exists() else None
    except Exception:
        return None


def _iter_individual_dirs(run_dir: Path) -> list[Path]:
    inds = [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("individual_")]
    return sorted(inds, key=_safe_mtime, reverse=True)


def _score_mp_individual(ind_dir: Path, mode: int) -> Optional[float]:
    mp = f"mp{mode}"
    try:
        true_n, pred_n, _t = _load_true_pred_time(ind_dir, mp)
        norm = _load_ma_norm(ind_dir)
        theta_true = true_n * norm
        theta_pred = pred_n * norm
        n = min(len(theta_true), len(theta_pred))
        if n <= 0:
            return None
        return _phase_percent_score(theta_true[:n], theta_pred[:n])
    except Exception:
        return None


def _score_sc_individual(ind_dir: Path, mode: int) -> Optional[float]:
    mps = f"mps{mode}"
    mpc = f"mpc{mode}"
    mps_dir = ind_dir / mps
    mpc_dir = ind_dir / mpc
    if not mps_dir.exists() or not mpc_dir.exists():
        # Some partial individuals may have only one side.
        return None
    try:
        # Results arrays are already in physical units; do NOT apply ma_norm here.
        sin_true, sin_pred, _t_sin = _load_true_pred_time(mps_dir, mps)
        cos_true, cos_pred, _t_cos = _load_true_pred_time(mpc_dir, mpc)

        n = min(len(sin_true), len(sin_pred), len(cos_true), len(cos_pred))
        if n <= 0:
            return None

        # Match the optimization metric: normalize (sin,cos) pairs to unit magnitude before atan2.
        ts, tc = _normalize_sincos_pairs(sin_true[:n], cos_true[:n])
        ps, pc = _normalize_sincos_pairs(sin_pred[:n], cos_pred[:n])
        theta_true = np.arctan2(ts, tc)
        theta_pred = np.arctan2(ps, pc)
        return _phase_percent_score(theta_true, theta_pred)
    except Exception:
        return None


def _pick_best_mp_individual(run_dir: Path, mode: int) -> Path:
    by_csv = _best_individual_from_csv(run_dir)
    if by_csv is not None:
        return by_csv

    best: tuple[float, Path] | None = None
    for ind in _iter_individual_dirs(run_dir):
        s = _score_mp_individual(ind, mode)
        if s is None:
            continue
        if best is None or s < best[0]:
            best = (s, ind)
    if best is None:
        raise FileNotFoundError(f"No usable mp{mode} individual found in {run_dir}")
    return best[1]


def _pick_best_sc_individual(run_dir: Path, mode: int, match: Optional[dict[str, Any]] = None) -> Path:
    """
    Pick the best mp_sc individual. If `match` is provided, try to match on keys:
    - state
    - reserved_shot
    - notebook_type
    """
    by_csv = _best_individual_from_csv(run_dir)
    if by_csv is not None and match:
        # If the CSV-best doesn't match, fall back to scoring with matching preference.
        p = _load_params(by_csv)
        keys = ("state", "reserved_shot", "notebook_type")
        if all(match.get(k) is None or p.get(k) == match.get(k) for k in keys):
            return by_csv

    best: tuple[float, Path] | None = None
    best_unmatched: tuple[float, Path] | None = None
    keys = ("state", "reserved_shot")

    for ind in _iter_individual_dirs(run_dir):
        s = _score_sc_individual(ind, mode)
        if s is None:
            continue
        if match:
            p = _normalize_params(_load_params(ind))
            if all(match.get(k) is None or p.get(k) == match.get(k) for k in keys):
                if best is None or s < best[0]:
                    best = (s, ind)
                continue
        if best_unmatched is None or s < best_unmatched[0]:
            best_unmatched = (s, ind)

    if best is not None:
        return best[1]
    if best_unmatched is not None:
        return best_unmatched[1]
    raise FileNotFoundError(f"No usable mp_sc{mode} individual found in {run_dir}")


def _pick_best_mp_individual_matching(run_dir: Path, mode: int, match: Optional[dict[str, Any]] = None) -> Path:
    """
    Pick the best mp individual in `run_dir`, preferring those matching `state` and `reserved_shot`.
    Falls back to the best overall in the run if no matches exist.
    """
    if not match:
        return _pick_best_mp_individual(run_dir, mode)

    match = _normalize_params(match)
    keys = ("state", "reserved_shot", "notebook_type")

    best: tuple[float, Path] | None = None
    best_unmatched: tuple[float, Path] | None = None

    for ind in _iter_individual_dirs(run_dir):
        s = _score_mp_individual(ind, mode)
        if s is None:
            continue
        p = _normalize_params(_load_params(ind))
        p_nt = _infer_notebook_type(ind)
        if p_nt is not None:
            p["notebook_type"] = p_nt
        if all(match.get(k) is None or p.get(k) == match.get(k) for k in keys):
            if best is None or s < best[0]:
                best = (s, ind)
        else:
            if best_unmatched is None or s < best_unmatched[0]:
                best_unmatched = (s, ind)

    if best is not None:
        return best[1]
    if best_unmatched is not None:
        return best_unmatched[1]
    raise FileNotFoundError(f"No usable mp{mode} individual found in {run_dir}")

def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _phase_percent_score(theta_true: np.ndarray, theta_pred: np.ndarray) -> float:
    delta = _wrap_to_pi(theta_pred - theta_true)
    return float(np.mean(np.abs(delta)) / math.pi * 100.0)


def _maybe_unwrap(theta: np.ndarray, unwrap: bool) -> np.ndarray:
    if not unwrap:
        return _wrap_to_pi(theta)
    # np.unwrap assumes phase changes are small between samples; reasonable for visual continuity.
    return np.unwrap(_wrap_to_pi(theta))


def _normalize_sincos_pairs(sin: np.ndarray, cos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sin = np.asarray(sin, dtype=float)
    cos = np.asarray(cos, dtype=float)
    n = np.sqrt(sin * sin + cos * cos)
    n = np.where(n < 1e-8, 1.0, n)
    return sin / n, cos / n


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Plot phase vs time: true vs base (mp) vs sin/cos reconstruction.")
    parser.add_argument(
        "--results_root",
        type=str,
        default="",
        help="Root directory containing optimization runs. Default: <repo_root>/data/optimization_results",
    )
    parser.add_argument("--mp_dir", type=str, default="", help="Directory containing results for mp<mode> (optional).")
    parser.add_argument("--mps_dir", type=str, default="", help="Directory containing results for mps<mode> (optional).")
    parser.add_argument("--mpc_dir", type=str, default="", help="Directory containing results for mpc<mode> (optional).")
    parser.add_argument("--mode", type=int, default=2, help="Mode index (1-4). Default: 2")
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output PNG path. Default: <mp_dir>/phase_comparison_mode<mode>.png",
    )
    parser.add_argument("--unwrap", action="store_true", help="Unwrap phases for a continuous plot.")
    parser.add_argument("--show", action="store_true", help="Display the plot window (in addition to saving).")
    args = parser.parse_args(argv)

    if args.mode not in (1, 2, 3, 4):
        raise SystemExit("--mode must be 1, 2, 3, or 4")

    repo_root = _repo_root_from_this_file()
    results_root = (
        Path(args.results_root).expanduser().resolve()
        if args.results_root
        else (repo_root / "data" / "optimization_results")
    )

    mp = f"mp{args.mode}"
    mps = f"mps{args.mode}"
    mpc = f"mpc{args.mode}"

    # If user provided mps/mpc directly, use them. Otherwise pick latest mp_sc run + best individual.
    if args.mps_dir and args.mpc_dir:
        mps_dir = Path(args.mps_dir).expanduser().resolve()
        mpc_dir = Path(args.mpc_dir).expanduser().resolve()
        sc_ind = None
    elif args.mps_dir or args.mpc_dir:
        raise SystemExit("Provide both --mps_dir and --mpc_dir, or neither (auto-pick).")
    else:
        sc_run = _match_latest_run(results_root, mode=args.mode, want="mp_sc")
        sc_ind = _pick_best_sc_individual(sc_run, mode=args.mode, match=None)
        mps_dir = sc_ind / mps
        mpc_dir = sc_ind / mpc

    # Auto-discover mp_dir if user didn't specify it.
    # Important: choose an mp individual that matches the *selected* mp_sc individual (same state/reserved shot),
    # otherwise you end up comparing different shots and the percent labels become meaningless.
    if args.mp_dir:
        mp_dir = Path(args.mp_dir).expanduser().resolve()
    else:
        mp_run = _match_latest_run(results_root, mode=args.mode, want="mp")
        match = _normalize_params(_load_params(sc_ind)) if sc_ind is not None else None
        # Also require the same notebook_type (trimmed vs untrimmed), inferred from mp_sc outputs.
        if match is not None and sc_ind is not None:
            sc_nt = _infer_notebook_type(mps_dir) or _infer_notebook_type(mpc_dir)
            if sc_nt is not None:
                match["notebook_type"] = sc_nt
        mp_dir = _pick_best_mp_individual_matching(mp_run, mode=args.mode, match=match)

    # Load normalized arrays (+ time from mp* by default).
    mp_true_n, mp_pred_n, t = _load_true_pred_time(mp_dir, mp)
    mps_true_n, mps_pred_n = _load_true_pred(mps_dir, mps)
    mpc_true_n, mpc_pred_n = _load_true_pred(mpc_dir, mpc)

    # Results arrays are already in physical units; do NOT apply ma_norm here.
    theta_true = mp_true_n
    theta_pred_base = mp_pred_n
    sin_true = mps_true_n
    sin_pred = mps_pred_n
    cos_true = mpc_true_n
    cos_pred = mpc_pred_n

    # Compute lengths for metrics (don't unnecessarily truncate).
    n_mp = min(len(theta_true), len(theta_pred_base))
    n_sc = min(len(sin_true), len(sin_pred), len(cos_true), len(cos_pred))
    if n_mp == 0 or n_sc == 0 or len(t) == 0:
        raise SystemExit("No overlapping samples to plot (arrays are empty).")

    # Plot length: align to a single timebase (mp time). In matched runs, these should be equal.
    n_plot = min(len(t), n_mp, n_sc)
    t = t[:n_plot]
    theta_true = theta_true[:n_plot]
    theta_pred_base = theta_pred_base[:n_plot]
    sin_true = sin_true[:n_plot]
    sin_pred = sin_pred[:n_plot]
    cos_true = cos_true[:n_plot]
    cos_pred = cos_pred[:n_plot]

    # Reconstruct phases from sin/cos (match optimization metric: normalize pairs first).
    st, ct = _normalize_sincos_pairs(sin_true, cos_true)
    sp, cp = _normalize_sincos_pairs(sin_pred, cos_pred)
    theta_true_sc = np.arctan2(st, ct)
    theta_pred_sc = np.arctan2(sp, cp)

    # Wrap/unwrap for display (do NOT change underlying error calculations).
    theta_true_plot = _maybe_unwrap(theta_true, args.unwrap)
    theta_base_plot = _maybe_unwrap(theta_pred_base, args.unwrap)
    theta_sc_plot = _maybe_unwrap(theta_pred_sc, args.unwrap)

    base_pct = _phase_percent_score(theta_true, theta_pred_base)
    # Use the same reference as the mp_sc optimizer (sin/cos -> theta_true_sc).
    sc_pct = _phase_percent_score(theta_true_sc, theta_pred_sc)

    # Default output location:
    # - If we auto-picked an mp_sc individual, save alongside it (that's the "winner" the user usually cares about).
    # - Otherwise (manual dirs), fall back to mp_dir.
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        out_base = sc_ind if sc_ind is not None else mp_dir
        out_path = out_base / f"phase_comparison_mode{args.mode}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Import matplotlib lazily so this script can be imported without display deps.
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))
    plt.plot(t, theta_true_plot, linewidth=2.0, label="True phase (mp)")
    plt.plot(t, theta_base_plot, linewidth=1.5, label=f"Base/raw model (mp pred) [{base_pct:.2f}%]")
    plt.plot(t, theta_sc_plot, linewidth=1.5, label=f"sin/cos -> atan2 pred [{sc_pct:.2f}%]")

    plt.xlabel("Time")
    plt.ylabel("Phase (rad)")
    title = f"Phase comparison (mode {args.mode})"
    if args.unwrap:
        title += " [unwrapped]"
    else:
        title += " [wrapped to [-pi, pi]]"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)

    if args.show:
        plt.show()
    else:
        plt.close()

    print(f"Auto-picked mp individual   : {mp_dir}")
    if sc_ind is not None:
        print(f"Auto-picked mp_sc individual: {sc_ind}")
    else:
        # User-provided mps/mpc paths; keep output brief but informative.
        print("Using user-provided mps/mpc directories.")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

