from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASIC_MODELS = [
    ("model1", "I (Drake)"),
    ("model2", "II (Reduced)"),
    ("model3a", "IIIA (Settlement)"),
    ("model3b", "IIIB-M"),
    ("model4", "IV (Rare-Earth)"),
]

SUPERMODELS = [
    ("SM1", "SM1"),
    ("SM2", "SM2"),
    ("SM3", "SM3"),
    ("SM4", "SM4"),
    ("SM5", "SM5"),
]


def _positive_finite(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones_like(arrays[0], dtype=bool)
    for arr in arrays:
        arr = np.asarray(arr, dtype=float)
        mask &= np.isfinite(arr) & (arr > 0.0)
    return mask


def _choose_nplot(data: np.lib.npyio.NpzFile, model_id: str) -> np.ndarray:
    if model_id in {"model3a", "model3b"} or model_id.startswith("SM"):
        if "N_site" not in data.files:
            raise KeyError(f"{model_id}: missing N_site in npz file.")
        return np.asarray(data["N_site"], dtype=float)

    if "N_active" not in data.files:
        raise KeyError(f"{model_id}: missing N_active in npz file.")
    return np.asarray(data["N_active"], dtype=float)


def compute_regime_masses(npz_path: Path, model_id: str) -> dict[str, float]:
    data = np.load(npz_path, allow_pickle=True)

    if "L_active" not in data.files:
        raise KeyError(f"{npz_path} does not contain L_active.")

    L = np.asarray(data["L_active"], dtype=float)
    N = _choose_nplot(data, model_id)

    if "weight" in data.files:
        w = np.asarray(data["weight"], dtype=float)
    else:
        w = np.ones_like(L, dtype=float)

    ok = _positive_finite(L, N)
    ok &= np.isfinite(w) & (w >= 0.0)

    L = L[ok]
    N = N[ok]
    w = w[ok]

    if L.size == 0:
        raise ValueError(f"No valid samples in {npz_path}.")

    if np.sum(w) <= 0.0:
        w = np.ones_like(L, dtype=float)

    r_exp = (L >= 1.0e4) & (N >= 1.0e2)
    r_short = L < 1.0e3
    r_medium = (L >= 1.0e3) & (L < 1.0e5) & (~r_exp)
    r_long_localized = (L >= 1.0e5) & (N < 1.0e2)

    total = float(np.sum(w))

    masses = {
        "Short": 100.0 * float(np.sum(w[r_short])) / total,
        "Medium": 100.0 * float(np.sum(w[r_medium])) / total,
        "Long-localized": 100.0 * float(np.sum(w[r_long_localized])) / total,
        "Multi-site": 100.0 * float(np.sum(w[r_exp])) / total,
    }

    check_sum = sum(masses.values())
    if abs(check_sum - 100.0) > 0.5:
        print(
            f"[WARN] Regime masses for {model_id} sum to {check_sum:.2f}%, "
        )

    return masses


def make_table(outdir: Path) -> pd.DataFrame:
    rows = []

    sample_dir = outdir / "samples"
    super_dir = outdir / "super_samples"

    for model_id, label in BASIC_MODELS:
        path = sample_dir / f"{model_id}.npz"
        masses = compute_regime_masses(path, model_id)
        rows.append({"Model": label, **masses})

    for model_id, label in SUPERMODELS:
        path = super_dir / f"{model_id}.npz"
        masses = compute_regime_masses(path, model_id)
        rows.append({"Model": label, **masses})

    return pd.DataFrame(rows)


def write_latex_table(df: pd.DataFrame, path: Path) -> None:
    def fmt(x: float) -> str:
        return f"{x:.1f}"

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Estimated conditional Monte Carlo mass by diagnostic regime (\%).}"
    )
    lines.append(r"\label{tab:regime_masses}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\small")
    lines.append(
        r"\begin{tabular}{"
        r">{\RaggedRight\arraybackslash}p{0.28\linewidth}"
        r">{\centering\arraybackslash}p{0.15\linewidth}"
        r">{\centering\arraybackslash}p{0.15\linewidth}"
        r">{\centering\arraybackslash}p{0.18\linewidth}"
        r">{\centering\arraybackslash}p{0.16\linewidth}"
        r"}"
    )
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Model} & \textbf{Short} & \textbf{Medium} & "
        r"\textbf{Long-localized} & \textbf{Multi-site}\\"
    )
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        lines.append(
            f"{row['Model']} & "
            f"{fmt(row['Short'])} & "
            f"{fmt(row['Medium'])} & "
            f"{fmt(row['Long-localized'])} & "
            f"{fmt(row['Multi-site'])}\\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs", help="Pipeline output directory.")
    args = parser.parse_args()

    outdir = Path(args.out)
    df = make_table(outdir)

    csv_path = outdir / "regime_masses_C2.csv"
    tex_path = outdir / "table_C2_regime_masses.tex"

    df.to_csv(csv_path, index=False)
    write_latex_table(df, tex_path)

    print(df.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print(f"[OK] wrote: {csv_path}")
    print(f"[OK] wrote: {tex_path}")


if __name__ == "__main__":
    main()