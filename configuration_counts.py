from __future__ import annotations

import argparse
import csv
from pathlib import Path

from triads import MODEL_NAMES, TRIADS


def compute(distributions: int = 3) -> tuple[list[dict[str, int]], int, int]:
    rows = []
    total = 0
    exponent = 0
    for model_id in MODEL_NAMES:
        n_parameters = len(TRIADS[model_id])
        configurations = distributions ** n_parameters
        rows.append({
            "model": model_id,
            "parameters": n_parameters,
            "configurations": configurations,
        })
        total += configurations
        exponent += n_parameters
    return rows, total, distributions ** exponent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributions", type=int, default=3)
    parser.add_argument("--out", default="outputs/configuration_counts.csv")
    args = parser.parse_args()
    rows, total, product = compute(args.distributions)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "parameters", "configurations"])
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({"model": "sum", "parameters": sum(row["parameters"] for row in rows), "configurations": total})
        writer.writerow({"model": "cartesian_product", "parameters": sum(row["parameters"] for row in rows), "configurations": product})
    print(f"native configurations: {total}")
    print(f"supermodel Cartesian product: {product}")
    print(f"[OK] wrote: {path}")


if __name__ == "__main__":
    main()
