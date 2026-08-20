from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")

from figure_style import configure_matplotlib_for_latex, savefig_publication
configure_matplotlib_for_latex()

import numpy as np
import matplotlib.pyplot as plt

from triads import MODEL_NAMES
from supermodels import SUPERMODEL_NAMES


BASIC_MODEL_DISPLAY = {
    "model1": "Model I",
    "model2": "Model II",
    "model3a": "Model IIIA",
    "model3b": "Model IIIB-M",
    "model4": "Model IV",
}

METRIC_DEFS = {
    "logL_active": {
        "field": "L_active",
        "transform": "log10",
        "label": r"$\log_{10}(L_{\mathrm{active}}/\mathrm{yr})$",
        "description": "Active lifetime",
    },
    "logL_surv": {
        "field": "L_surv",
        "transform": "log10",
        "label": r"$\log_{10}(L_{\mathrm{surv}}/\mathrm{yr})$",
        "description": "Survival-adjusted lifetime",
    },
    "logN_site": {
        "field": "N_site",
        "transform": "log10",
        "label": r"$\log_{10}(N_{\mathrm{site}})$",
        "description": "Active settled sites",
    },
    "logN_source": {
        "field": "N_source",
        "transform": "log10",
        "label": r"$\log_{10}(N_{\mathrm{source}})$",
        "description": "Independent source civilizations",
    },
    "cap_fraction": {
        "field": "cap_fraction",
        "transform": "identity",
        "label": r"$N_{\mathrm{site}}/N_{\mathrm{gal,cap}}$",
        "description": "Fraction of Galactic active-site cap",
    },
    "log_silence_gain": {
        "field": "silence_gain",
        "transform": "log10",
        "label": r"$\log_{10}(A_{\mathrm{sil}})$",
        "description": "Silence-survivorship factor",
    },
    "log_mu_obs": {
        "field": "mu_obs",
        "transform": "log10",
        "label": r"$\log_{10}(\mu_{\mathrm{obs}})$",
        "description": "Expected observed events",
    },
    "minus_log10_P_zero": {
        "field": "P_zero",
        "transform": "minus_log10",
        "label": r"$-\log_{10}P(K=0)$",
        "description": "Zero-event surprise",
    },
}

PRIMARY_METRICS = [
    "logL_active",
    "logN_site",
    "logN_source",
    "cap_fraction",
    "log_silence_gain",
    "log_mu_obs",
    "minus_log10_P_zero",
]


def _as_str_scalar(value) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return str(arr)


def _model_label(model_id: str) -> str:
    if model_id in BASIC_MODEL_DISPLAY:
        return BASIC_MODEL_DISPLAY[model_id]
    return model_id


def _safe_weight(data: np.lib.npyio.NpzFile, n: int) -> np.ndarray:
    if "weight" not in data.files:
        return np.ones(n, dtype=float)
    w = np.asarray(data["weight"], dtype=float)
    if w.shape[0] != n:
        return np.ones(n, dtype=float)
    ok = np.isfinite(w) & (w >= 0.0)
    if not np.any(ok) or float(np.sum(w[ok])) <= 0.0:
        return np.ones(n, dtype=float)
    w = np.where(ok, w, 0.0)
    return w


def weighted_quantile(x, w, qs):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    qs = np.asarray(qs, dtype=float)

    ok = np.isfinite(x) & np.isfinite(w) & (w >= 0.0)
    x = x[ok]
    w = w[ok]
    if x.size == 0:
        return np.full(qs.shape, np.nan, dtype=float)
    if float(np.sum(w)) <= 0.0:
        w = np.ones_like(x, dtype=float)

    order = np.argsort(x)
    x = x[order]
    w = w[order]
    cw = np.cumsum(w)
    cw = cw / cw[-1]
    return np.interp(qs, cw, x)


def weighted_mean(x, w) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(w) & (w >= 0.0)
    if not np.any(ok):
        return float("nan")
    if float(np.sum(w[ok])) <= 0.0:
        return float(np.mean(x[ok]))
    return float(np.sum(x[ok] * w[ok]) / np.sum(w[ok]))


def transform_metric(values: np.ndarray, transform: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if transform == "identity":
        return values
    if transform == "log10":
        return np.log10(np.maximum(values, 1.0e-300))
    if transform == "minus_log10":
        return -np.log10(np.clip(values, 1.0e-300, 1.0))
    raise ValueError(f"Unknown metric transform: {transform!r}")


def summarize_npz(path: Path, group: str, model_id: str | None = None) -> tuple[list[dict], dict[str, dict]]:
    with np.load(path, allow_pickle=False) as data:
        if model_id is None:
            if "model_id" in data.files:
                model_id = _as_str_scalar(data["model_id"])
            else:
                model_id = path.stem

        if "L_active" in data.files:
            n = int(np.asarray(data["L_active"]).shape[0])
        elif "N_site" in data.files:
            n = int(np.asarray(data["N_site"]).shape[0])
        else:
            return [], {}

        w = _safe_weight(data, n)
        rows: list[dict] = []
        metric_summaries: dict[str, dict] = {}

        for metric_name, spec in METRIC_DEFS.items():
            field = spec["field"]
            if field not in data.files:
                continue
            raw = np.asarray(data[field], dtype=float)
            if raw.shape[0] != n:
                continue

            values = transform_metric(raw, spec["transform"])
            ok = np.isfinite(values) & np.isfinite(w) & (w >= 0.0)
            if np.count_nonzero(ok) < 5:
                continue
            if not np.any(np.isfinite(values[ok])):
                continue

            q05, q10, q50, q90, q95 = weighted_quantile(values, w, [0.05, 0.10, 0.50, 0.90, 0.95])
            row = {
                "group": group,
                "model": model_id,
                "model_label": _model_label(model_id),
                "metric": metric_name,
                "description": spec["description"],
                "n": n,
                "sum_weight": float(np.sum(w[np.isfinite(w) & (w >= 0.0)])),
                "mean": weighted_mean(values, w),
                "q05": float(q05),
                "q10": float(q10),
                "q50": float(q50),
                "q90": float(q90),
                "q95": float(q95),
            }
            rows.append(row)
            metric_summaries[metric_name] = row

        return rows, metric_summaries


def _write_rows_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_wide_csv(path: Path, summaries: dict[str, dict[str, dict]]) -> None:
    metric_names = list(METRIC_DEFS.keys())
    rows = []
    for model_id, metrics in summaries.items():
        row: dict[str, object] = {
            "model": model_id,
            "model_label": _model_label(model_id),
        }
        for metric in metric_names:
            if metric not in metrics:
                continue
            for suffix in ("q05", "q50", "q95", "mean"):
                row[f"{metric}_{suffix}"] = metrics[metric][suffix]
        rows.append(row)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    fieldnames = ["model", "model_label"] + [f for f in fieldnames if f not in {"model", "model_label"}]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _collect_group(sample_dir: Path, model_ids: Iterable[str], group: str) -> tuple[list[dict], dict[str, dict]]:
    rows: list[dict] = []
    summaries: dict[str, dict] = {}
    for model_id in model_ids:
        path = sample_dir / f"{model_id}.npz"
        if not path.exists():
            continue
        model_rows, metric_summaries = summarize_npz(path, group=group, model_id=model_id)
        if model_rows:
            rows.extend(model_rows)
            summaries[model_id] = metric_summaries
    return rows, summaries


def _plot_interval(metric: str, summaries: dict[str, dict], out_path: Path) -> bool:
    spec = METRIC_DEFS[metric]
    items = []
    for model_id, metrics in summaries.items():
        if metric not in metrics:
            continue
        row = metrics[metric]
        q05 = row["q05"]
        q50 = row["q50"]
        q95 = row["q95"]
        if not (np.isfinite(q05) and np.isfinite(q50) and np.isfinite(q95)):
            continue
        items.append((_model_label(model_id), q05, q50, q95))

    if not items:
        return False

    labels = [item[0] for item in items]
    q05 = np.array([item[1] for item in items], dtype=float)
    q50 = np.array([item[2] for item in items], dtype=float)
    q95 = np.array([item[3] for item in items], dtype=float)
    y = np.arange(len(items), dtype=float)

    fig_h = max(2.2, 0.42 * len(items) + 0.8)
    fig, ax = plt.subplots(figsize=(6.2, fig_h))
    xerr = np.vstack([q50 - q05, q95 - q50])
    ax.errorbar(q50, y, xerr=xerr, fmt="o", capsize=3.0, linewidth=1.2, markersize=4.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(spec["label"])
    ax.grid(True, axis="x", alpha=0.25)
    ax.grid(False, axis="y")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    savefig_publication(fig, out_path)
    plt.close(fig)
    return True


def _plot_all_metrics(group_name: str, summaries: dict[str, dict], out_dir: Path, metrics: list[str] | None = None) -> list[Path]:
    if metrics is None:
        metrics = PRIMARY_METRICS
    written: list[Path] = []
    for metric in metrics:
        out_path = out_dir / f"{group_name}_{metric}_interval.pdf"
        if _plot_interval(metric, summaries, out_path):
            written.append(out_path)
    return written


def _plot_median_scatter(summaries: dict[str, dict], out_path: Path, *, group_name: str) -> bool:
    items = []
    for model_id, metrics in summaries.items():
        if "logL_active" not in metrics or "logN_site" not in metrics:
            continue
        x = metrics["logL_active"]["q50"]
        y = metrics["logN_site"]["q50"]
        if np.isfinite(x) and np.isfinite(y):
            items.append((_model_label(model_id), float(x), float(y)))
    if not items:
        return False

    fig, ax = plt.subplots(figsize=(5.4, 4.3))
    xs = [i[1] for i in items]
    ys = [i[2] for i in items]
    ax.scatter(xs, ys, s=36)
    for label, x, y in items:
        ax.annotate(label, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.set_xlabel(r"Median $\log_{10}(L_{\mathrm{active}}/\mathrm{yr})$")
    ax.set_ylabel(r"Median $\log_{10}(N_{\mathrm{site}})$")
    ax.grid(True, alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    savefig_publication(fig, out_path)
    plt.close(fig)
    return True


def run_result_summary(outdir: Path) -> None:
    outdir = Path(outdir)
    summary_dir = outdir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    all_summaries: dict[str, dict] = {}

    basic_rows, basic_summaries = _collect_group(outdir / "samples", MODEL_NAMES, group="basic")
    if basic_rows:
        all_rows.extend(basic_rows)
        all_summaries.update(basic_summaries)
        _write_rows_csv(summary_dir / "basic_summary_long.csv", basic_rows)
        _write_wide_csv(summary_dir / "basic_summary_wide.csv", basic_summaries)
        _plot_all_metrics("basic", basic_summaries, summary_dir)
        _plot_median_scatter(basic_summaries, summary_dir / "basic_median_L_vs_N_site.pdf", group_name="basic")

    super_rows, super_summaries = _collect_group(outdir / "super_samples", SUPERMODEL_NAMES, group="super")
    if super_rows:
        all_rows.extend(super_rows)
        all_summaries.update(super_summaries)
        _write_rows_csv(summary_dir / "super_summary_long.csv", super_rows)
        _write_wide_csv(summary_dir / "super_summary_wide.csv", super_summaries)
        _plot_all_metrics("super", super_summaries, summary_dir)
        _plot_median_scatter(super_summaries, summary_dir / "super_median_L_vs_N_site.pdf", group_name="super")

    if all_rows:
        _write_rows_csv(summary_dir / "all_summary_long.csv", all_rows)
        _write_wide_csv(summary_dir / "all_summary_wide.csv", all_summaries)
        _plot_all_metrics("all", all_summaries, summary_dir)
        _plot_median_scatter(all_summaries, summary_dir / "all_median_L_vs_N_site.pdf", group_name="all")
        print(f"[OK] summary outputs: {summary_dir}")
    else:
        print("[WARN] summary: no sample .npz files were found. Run generate/super before summary, or use --keep-npz from an earlier run.")
