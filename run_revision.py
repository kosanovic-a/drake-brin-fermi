from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("[RUN]", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("actions", nargs="+", choices=["counts", "table3", "observation", "semantic", "variance", "clustering"])
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--samples", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    python = sys.executable
    root = Path(args.out)
    for action in args.actions:
        if action == "counts":
            run([python, "configuration_counts.py", "--out", str(root / "revision" / "configuration_counts.csv")])
        elif action == "table3":
            run([python, "table3_robustness.py", "--samples", str(min(args.samples, 100000)), "--out", str(root / "revision" / "table3")])
        elif action == "observation":
            run([python, "run_observation_informed.py", "--out", str(root / "revision" / "observation"), "--samples", str(args.samples), "--seed", str(args.seed)])
        elif action == "semantic":
            run([python, "semantic_bridge_bma.py", "--sample-dir", str(root / "samples"), "--out-dir", str(root / "revision" / "semantic_bma")])
        elif action == "variance":
            run([python, "layered_variance.py", "--sample-dir", str(root / "samples"), "--out", str(root / "revision" / "layered_variance")])
        elif action == "clustering":
            run([python, "clustering_sensitivity.py", "--out", str(root), "--seed", str(args.seed)])


if __name__ == "__main__":
    main()
