from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from models import MODELS
from sampler import sample_params
from triads import TRIADS

PARAMETERS = ("fl", "fi", "fa")
SEEDS = (12345, 23456, 34567, 45678, 56789)


def transform(triads: dict[str, list[float]], mode: str) -> dict[str, list[float]]:
    result = {key: list(map(float, value)) for key, value in triads.items()}
    for name in PARAMETERS:
        lo, peak, hi = np.log10(result[name])
        if mode == "narrow":
            lo, hi = peak + 0.75 * (lo - peak), peak + 0.75 * (hi - peak)
        elif mode == "wide":
            lo, hi = peak + 1.25 * (lo - peak), peak + 1.25 * (hi - peak)
        elif mode == "peak_low":
            peak = peak + 0.25 * (lo - peak)
        elif mode == "peak_high":
            peak = peak + 0.25 * (hi - peak)
        values = 10.0 ** np.array([lo, peak, hi])
        values[2] = min(values[2], 1.0)
        values[1] = min(values[1], values[2])
        result[name] = values.tolist()
    return result


def region_masses(lifetime: np.ndarray, abundance: np.ndarray) -> tuple[float, float]:
    short = lifetime < 1.0e3
    high = (lifetime >= 1.0e4) & (abundance >= 1.0e2)
    long_localized = (lifetime >= 1.0e5) & (abundance < 1.0e2)
    return 100.0 * short.mean(), 100.0 * long_localized.mean()


def summarize(lifetime: np.ndarray, abundance: np.ndarray) -> dict[str, float]:
    log_lifetime = np.log10(lifetime)
    q05, q50, q95 = np.quantile(log_lifetime, [0.05, 0.50, 0.95])
    short, long_localized = region_masses(lifetime, abundance)
    return {
        "median": float(q50),
        "w90": float(q95 - q05),
        "short": short,
        "long_localized": long_localized,
    }


def run(samples: int, seeds: tuple[int, ...], outdir: Path) -> pd.DataFrame:
    scenarios = (
        ("baseline", "baseline", "lognormal"),
        ("bounds_narrowed_25", "narrow", "lognormal"),
        ("bounds_widened_25", "wide", "lognormal"),
        ("peaks_lower", "peak_low", "lognormal"),
        ("peaks_upper", "peak_high", "lognormal"),
        ("loglinear", "baseline", "loglinear"),
        ("gaussian", "baseline", "gauss"),
    )
    records = []
    for seed in seeds:
        for name, mode, distribution in scenarios:
            rng = np.random.default_rng(seed)
            triads = TRIADS["model1"] if mode == "baseline" else transform(TRIADS["model1"], mode)
            params = sample_params(triads, samples, distribution, rng)
            output = MODELS["model1"](params)
            records.append({"seed": seed, "scenario": name, **summarize(output["L_active"], output["N_active"])})
    raw = pd.DataFrame(records)
    baseline = raw[raw["scenario"] == "baseline"].set_index("seed")
    rows = []
    for scenario, group in raw.groupby("scenario", sort=False):
        joined = group.set_index("seed").join(baseline[["median", "w90"]], rsuffix="_base")
        rows.append({
            "scenario": scenario,
            "delta_median": float((joined["median"] - joined["median_base"]).mean()),
            "w90_ratio": float((joined["w90"] / joined["w90_base"]).mean()),
            "P_short_percent": float(group["short"].mean()),
            "P_long_localized_percent": float(group["long_localized"].mean()),
            "between_seed_sd_delta_median": float((joined["median"] - joined["median_base"]).std(ddof=1)),
        })
    outdir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(outdir / "table3_seed_results.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "table3_robustness.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--out", default="outputs/revision/table3")
    args = parser.parse_args()
    summary = run(args.samples, SEEDS, Path(args.out))
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
