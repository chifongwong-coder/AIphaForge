"""v2.2.1 #6: Generate the precomputed Spearman null-quantile table.

Output: ``src/aiphaforge/probes/_rank_null.npz``.

Layout per v2.2.1 r7 §6:
- N ∈ [3, 12]: exact enumeration of all N! permutations.
- N ∈ [13, 20]: Monte Carlo with 10⁶ samples per N, PCG64-seeded
  via ``np.random.SeedSequence(MC_ROOT_SEED).spawn(N_RANGE_LEN)``
  so per-N regeneration isolation works.

Per-N storage is the sorted absolute-ρ array. Total file size
~few MB. Loaded once via ``np.load`` and cached in module state.

Run from repo root:

    python scripts/generate_rank_null_table.py

This script is committed for audit; the .npz it produces is the
runtime artifact.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
from itertools import permutations
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "src/aiphaforge/probes/_rank_null.npz"

EXACT_N_RANGE = (3, 10)         # inclusive — 10! = 3.6M, ~seconds
MC_N_RANGE = (11, 20)           # inclusive — 10⁶ samples each
# v2.2.1 #6 NOTE: r7 plan said exact ≤ 12, but 12! = 480M
# permutations is impractical at pure-Python iteration speeds.
# MC at 10⁶ samples gives SE ≈ 1e-3 at q=0.99 which is the
# calibrated tolerance the cutoff lookup needs. The MC range
# extending to N=11 is a documented practical choice.
MC_SAMPLES = 1_000_000          # per N
MC_ROOT_SEED = 20260512         # locked v2.2.1
GENERATOR_VERSION = "v2.2.1"


def tie_corrected_spearman(
    x_ranks: np.ndarray, y_ranks: np.ndarray,
) -> float:
    """Same formula as src/aiphaforge/probes/_rank.py — duplicated
    here so this script has no runtime dependency on the package
    being importable.
    """
    n = len(x_ranks)
    if n < 2:
        return 0.0

    def _tie_term(ranks: np.ndarray) -> float:
        unique, counts = np.unique(ranks, return_counts=True)
        return float(
            np.sum((counts.astype(float) ** 3) - counts.astype(float))
        )

    d = x_ranks.astype(float) - y_ranks.astype(float)
    sum_d2 = float(np.sum(d ** 2))
    n3_n = float(n ** 3 - n)
    t_x = _tie_term(x_ranks)
    t_y = _tie_term(y_ranks)
    numerator = n3_n - 6.0 * sum_d2 - 0.5 * (t_x + t_y)
    denom = math.sqrt(max((n3_n - t_x), 0.0) * max((n3_n - t_y), 0.0))
    if denom == 0:
        return 0.0
    return numerator / denom


def _exact_enumerate_abs_rho(n: int) -> np.ndarray:
    """All N! permutations; return sorted |ρ| array."""
    canonical = np.arange(1, n + 1, dtype=float)
    samples = []
    for perm in permutations(canonical):
        samples.append(
            abs(tie_corrected_spearman(canonical, np.asarray(perm)))
        )
    arr = np.asarray(samples, dtype=np.float64)
    arr.sort()
    return arr


def _mc_abs_rho(n: int, n_samples: int, seed: int) -> np.ndarray:
    """Monte Carlo: n_samples random permutations; return sorted
    |ρ| array.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    canonical = np.arange(1, n + 1, dtype=float)
    samples = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        shuf = rng.permutation(canonical)
        samples[i] = abs(tie_corrected_spearman(canonical, shuf))
    samples.sort()
    return samples


def main(out_path: Path = OUT_PATH) -> None:
    print(f"[v2.2.1 #6] Generating rank-null table → {out_path}")
    arrays: dict[str, np.ndarray] = {}

    # Exact enumeration for N=3..12.
    for n in range(EXACT_N_RANGE[0], EXACT_N_RANGE[1] + 1):
        print(f"  N={n}: exact enumeration ({math.factorial(n)} perms)...")
        arrays[f"abs_rho_n{n}"] = _exact_enumerate_abs_rho(n)

    # Monte Carlo for N=13..20 with per-N derived seeds.
    n_mc_range = MC_N_RANGE[1] - MC_N_RANGE[0] + 1
    seed_seq = np.random.SeedSequence(MC_ROOT_SEED)
    spawned = seed_seq.spawn(n_mc_range)
    per_n_seeds = []
    for offset, sub_seed in enumerate(spawned):
        n = MC_N_RANGE[0] + offset
        # Derive a single int seed for PCG64 from the SeedSequence.
        derived = int(sub_seed.generate_state(1, dtype=np.uint64)[0])
        per_n_seeds.append(derived)
        print(
            f"  N={n}: MC ({MC_SAMPLES} samples, seed={derived})..."
        )
        arrays[f"abs_rho_n{n}"] = _mc_abs_rho(n, MC_SAMPLES, derived)

    # Metadata
    arrays["__meta_mc_root_seed"] = np.asarray([MC_ROOT_SEED], dtype=np.int64)
    arrays["__meta_mc_per_n_seeds"] = np.asarray(
        per_n_seeds, dtype=np.uint64
    )
    arrays["__meta_exact_n_range"] = np.asarray(
        EXACT_N_RANGE, dtype=np.int32
    )
    arrays["__meta_mc_n_range"] = np.asarray(
        MC_N_RANGE, dtype=np.int32
    )
    arrays["__meta_generated_at_iso"] = np.asarray(
        [dt.datetime.now(dt.timezone.utc).isoformat()],
    )
    arrays["__meta_generator_version"] = np.asarray(
        [GENERATOR_VERSION],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    print(f"[v2.2.1 #6] Wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=OUT_PATH,
        help="output .npz path",
    )
    args = parser.parse_args()
    main(args.out)
