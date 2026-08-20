from __future__ import annotations

import argparse
from pathlib import Path

from cluster_validation import run_cluster_validation
from regime_sensitivity import run_regime_sensitivity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--max-rows", type=int, default=30000)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--skip-regime-sensitivity", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.out)
    run_cluster_validation(
        outdir,
        max_rows=args.max_rows,
        k_min=args.k_min,
        k_max=args.k_max,
        seed=args.seed,
    )
    if not args.skip_regime_sensitivity:
        run_regime_sensitivity(outdir)
    print(f"[OK] clustering diagnostics: {(outdir / 'clustering').resolve()}")


if __name__ == "__main__":
    main()
