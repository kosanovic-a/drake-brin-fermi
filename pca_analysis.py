from __future__ import annotations

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")

from figure_style import configure_matplotlib_for_latex, savefig_publication
configure_matplotlib_for_latex()


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = False
from matplotlib.patches import Ellipse

from triads import MODEL_NAMES
from supermodels import SUPERMODEL_NAMES


MODEL_LABELS = {
    "model1": "I",
    "model2": "II",
    "model3a": "IIIA",
    "model3b": "IIIB-M",
    "model4": "IV",
}


FEATURE_NAMES = [
    "logL_active",
    "logN_site",
    "logN_source",
    "logN_site_over_source",
    "cap_fraction",
]

FEATURE_LABELS = {
    "logL_active": r"$\log_{10} L_{\mathrm{active}}$",
    "logN_site": r"$\log_{10} N_{\mathrm{site}}$",
    "logN_source": r"$\log_{10} N_{\mathrm{source}}$",
    "logN_site_over_source": r"$\log_{10}(N_{\mathrm{site}}/N_{\mathrm{source}})$",
    "cap_fraction": r"$N_{\mathrm{site}}/N_{\mathrm{gal,cap}}$",
}


def _as_str_scalar(x) -> str:
    arr = np.asarray(x)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return str(arr)


def _log10_positive(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan, dtype=float)
    ok = np.isfinite(x) & (x > 0.0)
    out[ok] = np.log10(x[ok])
    return out


def _optional_array(data, key: str, shape, default: float = np.nan) -> np.ndarray:
    if key not in data.files:
        return np.full(shape, default, dtype=float)
    arr = np.asarray(data[key], dtype=float)
    if arr.shape != shape:
        return np.full(shape, default, dtype=float)
    return arr


def _sample_indices(weight: np.ndarray, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    n = int(weight.size)
    max_rows = int(max(1, max_rows))
    if n <= max_rows:
        return np.arange(n, dtype=int)

    w = np.asarray(weight, dtype=float)
    ok = np.isfinite(w) & (w >= 0.0)
    if not np.any(ok) or float(np.sum(w[ok])) <= 0.0:
        p = None
    else:
        w2 = np.where(ok, w, 0.0)
        p = w2 / np.sum(w2)

    return rng.choice(n, size=max_rows, replace=False, p=p)


def _load_sample_rows(
    npz_path: Path,
    *,
    max_rows: int,
    rng: np.random.Generator,
    lifetime_key: str = "L_active",
    fixed_group: str | None = None,
):
    
    data = np.load(npz_path, allow_pickle=True)

    required = [lifetime_key, "N_site", "N_source"]
    for key in required:
        if key not in data.files:
            raise KeyError(f"{npz_path} does not contain required key {key!r}.")

    L = np.asarray(data[lifetime_key], dtype=float)
    N_site = np.asarray(data["N_site"], dtype=float)
    N_source = np.asarray(data["N_source"], dtype=float)
    shape = L.shape

    if "weight" in data.files:
        weight = np.asarray(data["weight"], dtype=float)
    else:
        weight = np.ones(shape, dtype=float)

    if weight.shape != shape:
        weight = np.ones(shape, dtype=float)

    logL = _log10_positive(L)
    logN_site = _log10_positive(N_site)
    logN_source = _log10_positive(N_source)
    log_ratio = logN_site - logN_source

    cap_fraction = _optional_array(data, "cap_fraction", shape, default=0.0)
    cap_fraction = np.where(np.isfinite(cap_fraction), cap_fraction, 0.0)
    cap_fraction = np.clip(cap_fraction, 0.0, 1.0)

    X = np.column_stack([logL, logN_site, logN_source, log_ratio, cap_fraction])
    ok = np.all(np.isfinite(X), axis=1)
    ok &= np.isfinite(weight) & (weight >= 0.0)

    if not np.any(ok):
        raise ValueError(f"No valid finite PCA rows in {npz_path}.")

    X = X[ok]
    weight = weight[ok]

    if "component_model" in data.files:
        component = np.asarray(data["component_model"])[ok].astype(str)
    else:
        model_id = _as_str_scalar(data["model_id"] if "model_id" in data.files else npz_path.stem)
        component = np.full(X.shape[0], model_id, dtype="U16")

    group = fixed_group
    if group is None:
        group = _as_str_scalar(data["model_id"] if "model_id" in data.files else npz_path.stem)
    group_labels = np.full(X.shape[0], group, dtype="U32")

    idx = _sample_indices(weight, max_rows=max_rows, rng=rng)
    return {
        "X": X[idx],
        "weight": weight[idx],
        "group": group_labels[idx],
        "component": component[idx],
        "source_file": np.full(idx.size, npz_path.stem, dtype="U64"),
    }


def _concat_rows(row_sets: list[dict]) -> dict:
    if not row_sets:
        raise ValueError("No PCA row sets were provided.")
    keys = row_sets[0].keys()
    return {key: np.concatenate([rs[key] for rs in row_sets]) for key in keys}


def _robust_standardize(X: np.ndarray):
    X = np.asarray(X, dtype=float)
    lo = np.nanquantile(X, 0.005, axis=0)
    hi = np.nanquantile(X, 0.995, axis=0)
    Xw = np.clip(X, lo, hi)

    center = np.nanmedian(Xw, axis=0)
    q25 = np.nanquantile(Xw, 0.25, axis=0)
    q75 = np.nanquantile(Xw, 0.75, axis=0)
    scale = (q75 - q25) / 1.349

    std = np.nanstd(Xw, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1.0e-12), scale, std)
    scale = np.where(np.isfinite(scale) & (scale > 1.0e-12), scale, 1.0)

    min_scale = np.array([0.25, 0.25, 0.25, 0.25, 0.25], dtype=float)
    scale = np.maximum(scale, min_scale)

    Z = (Xw - center) / scale
    return Z, {"lo": lo, "hi": hi, "center": center, "scale": scale}


def _pca_scores(Z: np.ndarray, n_components: int = 3):
    if Z.ndim != 2 or Z.shape[0] < 2:
        raise ValueError("PCA requires at least two rows.")

    Zc = Z - np.mean(Z, axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    k = min(n_components, U.shape[1])
    scores = U[:, :k] * S[:k]

    if k < n_components:
        scores = np.column_stack([scores, np.zeros((scores.shape[0], n_components - k))])

    denom = max(1, Z.shape[0] - 1)
    eigvals = (S**2) / denom
    total = float(np.sum(eigvals))
    if total > 0.0 and np.isfinite(total):
        explained = eigvals[:k] / total
    else:
        explained = np.zeros(k)

    if k < n_components:
        explained = np.concatenate([explained, np.zeros(n_components - k)])

    loadings = Vt[:k].T
    if k < n_components:
        loadings = np.column_stack([loadings, np.zeros((loadings.shape[0], n_components - k))])

    return scores, explained, loadings


def _kmeans_pp_init(Y: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = Y.shape[0]
    centers = np.empty((k, Y.shape[1]), dtype=float)
    centers[0] = Y[rng.integers(0, n)]
    closest_d2 = np.sum((Y - centers[0]) ** 2, axis=1)

    for c in range(1, k):
        total = float(np.sum(closest_d2))
        if total <= 0.0 or not np.isfinite(total):
            centers[c] = Y[rng.integers(0, n)]
        else:
            centers[c] = Y[rng.choice(n, p=closest_d2 / total)]
        d2 = np.sum((Y - centers[c]) ** 2, axis=1)
        closest_d2 = np.minimum(closest_d2, d2)

    return centers


def _kmeans(Y: np.ndarray, n_clusters: int, rng: np.random.Generator, n_init: int = 10, max_iter: int = 120):
    n = Y.shape[0]
    k = int(max(1, min(n_clusters, n)))

    best_labels = None
    best_centers = None
    best_inertia = np.inf

    for _ in range(n_init):
        centers = _kmeans_pp_init(Y, k, rng)
        labels = np.zeros(n, dtype=int)

        for it in range(max_iter):
            d2 = np.sum((Y[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            new_labels = np.argmin(d2, axis=1)
            if np.array_equal(new_labels, labels) and it > 0:
                break
            labels = new_labels
            for c in range(k):
                mask = labels == c
                if np.any(mask):
                    centers[c] = np.mean(Y[mask], axis=0)
                else:
                    centers[c] = Y[rng.integers(0, n)]

        inertia = float(np.sum((Y - centers[labels]) ** 2))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    return best_labels, best_centers


def _axis_label(i: int, explained: np.ndarray) -> str:
    pct = 100.0 * float(explained[i]) if i < explained.size else 0.0
    return f"PC{i + 1} ({pct:.1f}\\%)"


def _ordered_unique(labels) -> list[str]:
    vals = list(dict.fromkeys(np.asarray(labels).astype(str).tolist()))
    if vals and all(v.isdigit() for v in vals):
        return sorted(vals, key=lambda x: int(x))
    return vals


def _color_lookup(labels, *, cluster: bool = False):
    unique = _ordered_unique(labels)
    if cluster:
        cmap = plt.get_cmap("tab20")
    else:
        cmap = plt.get_cmap("tab10") if len(unique) <= 10 else plt.get_cmap("tab20")
    return {lab: cmap(i % cmap.N) for i, lab in enumerate(unique)}


def _add_ellipse(ax, pts: np.ndarray, color):
    if pts.shape[0] < 8:
        return
    cov = np.cov(pts[:, :2].T)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    if vals[0] <= 0.0:
        return
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2.0 * np.sqrt(vals[:2])
    center = np.mean(pts[:, :2], axis=0)
    ell = Ellipse(
        xy=center,
        width=width,
        height=height,
        angle=angle,
        facecolor="none",
        edgecolor=color,
        lw=0.8,
        alpha=0.7,
    )
    ax.add_patch(ell)



def _weighted_centroid(pts: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.zeros(pts.shape[1] if pts.ndim == 2 else 0, dtype=float)

    if weights is None:
        return np.mean(pts, axis=0)

    w = np.asarray(weights, dtype=float).reshape(-1)
    ok = np.isfinite(w) & (w >= 0.0)
    if w.size != pts.shape[0] or not np.any(ok) or float(np.sum(w[ok])) <= 0.0:
        return np.mean(pts, axis=0)

    w = np.where(ok, w, 0.0)
    return np.average(pts, axis=0, weights=w)


def _overlay_centroids_2d(ax, scores: np.ndarray, labels, weights, colors, *, cluster_panel: bool = False):
    labels = np.asarray(labels).astype(str)
    weights = None if weights is None else np.asarray(weights, dtype=float)

    for lab in _ordered_unique(labels):
        mask = labels == lab
        if not np.any(mask):
            continue
        cen = _weighted_centroid(scores[mask, :2], None if weights is None else weights[mask])
        edge = 'black'
        face = 'white'
        ax.scatter(cen[0], cen[1], s=145 if cluster_panel else 115, marker='o',
                   facecolors=face, edgecolors=edge, linewidths=1.0,
                   alpha=0.98, zorder=30)
        ax.scatter(cen[0], cen[1], s=82 if cluster_panel else 66, marker='X',
                   color=colors[lab], edgecolors='white', linewidths=0.55,
                   alpha=1.0, zorder=31)


def _overlay_centroids_3d(ax, scores: np.ndarray, labels, weights, colors, *, cluster_panel: bool = False):
    labels = np.asarray(labels).astype(str)
    weights = None if weights is None else np.asarray(weights, dtype=float)
    zmin = float(np.nanmin(scores[:, 2]))
    zmax = float(np.nanmax(scores[:, 2]))
    zlift = 0.025 * max(1.0, zmax - zmin)

    for lab in _ordered_unique(labels):
        mask = labels == lab
        if not np.any(mask):
            continue
        cen = _weighted_centroid(scores[mask, :3], None if weights is None else weights[mask])
        z = float(cen[2] + zlift)
        ax.scatter([cen[0]], [cen[1]], [z], s=160 if cluster_panel else 125, marker='o',
                   facecolors='white', edgecolors='black', linewidths=1.05,
                   alpha=0.99, depthshade=False, zorder=100)
        ax.scatter([cen[0]], [cen[1]], [z], s=92 if cluster_panel else 74, marker='X',
                   color=colors[lab], edgecolors='white', linewidths=0.55,
                   alpha=1.0, depthshade=False, zorder=101)



def _scatter_panel(
    ax,
    scores: np.ndarray,
    labels,
    explained: np.ndarray,
    *,
    legend_title: str,
    cluster_panel: bool = False,
    centroid_labels: bool = False,
    ellipses: bool = False,
    show_legend: bool = True,
    weights=None,
):
    labels = np.asarray(labels).astype(str)
    colors = _color_lookup(labels, cluster=cluster_panel)

    for lab in _ordered_unique(labels):
        mask = labels == lab
        ax.scatter(
            scores[mask, 0],
            scores[mask, 1],
            s=5.5,
            alpha=0.38,
            color=colors[lab],
            edgecolors="none",
            rasterized=True,
            label=lab,
        )
        if ellipses:
            _add_ellipse(ax, scores[mask, :2], colors[lab])

    ax.axhline(0.0, color="0.85", lw=0.5, zorder=0)
    ax.axvline(0.0, color="0.85", lw=0.5, zorder=0)
    ax.set_xlabel(_axis_label(0, explained))
    ax.set_ylabel(_axis_label(1, explained))
    ax.grid(True, color="0.90", lw=0.4)

    if centroid_labels or cluster_panel:
        _overlay_centroids_2d(ax, scores, labels, weights, colors, cluster_panel=cluster_panel)

    if show_legend and len(_ordered_unique(labels)) > 1:
        leg = ax.legend(title=legend_title, fontsize=7, title_fontsize=7, frameon=False, loc="best", markerscale=1.8)
        for h in getattr(leg, "legend_handles", []):
            try:
                h.set_alpha(1.0)
            except Exception:
                pass


def _scatter_panel_3d(
    ax,
    scores: np.ndarray,
    labels,
    explained: np.ndarray,
    *,
    legend_title: str,
    cluster_panel: bool = False,
    centroid_labels: bool = False,
    show_legend: bool = True,
    view_elev: float = 28.0,
    view_azim: float = -60.0,
    weights=None,
):
    labels = np.asarray(labels).astype(str)
    colors = _color_lookup(labels, cluster=cluster_panel)

    for lab in _ordered_unique(labels):
        mask = labels == lab
        ax.scatter(
            scores[mask, 0],
            scores[mask, 1],
            scores[mask, 2],
            s=5.2,
            alpha=0.35,
            color=colors[lab],
            edgecolors="none",
            depthshade=False,
            rasterized=True,
            label=lab,
        )

    ax.set_xlabel(_axis_label(0, explained), labelpad=6)
    ax.set_ylabel(_axis_label(1, explained), labelpad=6)
    ax.set_zlabel(_axis_label(2, explained), labelpad=6)
    ax.view_init(elev=float(view_elev), azim=float(view_azim))
    ax.grid(True, color="0.90", lw=0.4)


    try:
        ax.set_box_aspect((1.15, 1.0, 0.80))
    except Exception:
        pass

    if centroid_labels or cluster_panel:
        pass

    if show_legend and len(_ordered_unique(labels)) > 1:
        leg = ax.legend(title=legend_title, fontsize=7, title_fontsize=7, frameon=False, loc="best", markerscale=1.9)
        for h in getattr(leg, "legend_handles", []):
            try:
                h.set_alpha(1.0)
            except Exception:
                pass


def _plot_single_pca_2d(
    scores: np.ndarray,
    explained: np.ndarray,
    labels,
    weights,
    out_path: Path,
    *,
    legend_title: str,
    cluster_panel: bool = False,
    centroid_labels: bool = False,
    ellipses: bool = False,
    show_legend: bool = True,
):
    fig, ax = plt.subplots(figsize=(4.15, 3.45))
    _scatter_panel(
        ax,
        scores,
        labels,
        explained,
        legend_title=legend_title,
        cluster_panel=cluster_panel,
        centroid_labels=centroid_labels,
        ellipses=ellipses,
        show_legend=show_legend,
        weights=weights,
    )
    fig.tight_layout(pad=0.7)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig_publication(fig, out_path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def _plot_single_pca_3d(
    scores: np.ndarray,
    explained: np.ndarray,
    labels,
    weights,
    out_path: Path,
    *,
    legend_title: str,
    cluster_panel: bool = False,
    centroid_labels: bool = False,
    show_legend: bool = True,
    view_elev: float = 28.0,
    view_azim: float = -60.0,
):
    fig = plt.figure(figsize=(4.65, 3.65))
    ax = fig.add_subplot(111, projection="3d")
    _scatter_panel_3d(
        ax,
        scores,
        labels,
        explained,
        legend_title=legend_title,
        cluster_panel=cluster_panel,
        centroid_labels=centroid_labels,
        show_legend=show_legend,
        view_elev=view_elev,
        view_azim=view_azim,
        weights=weights,
    )
    fig.tight_layout(pad=0.45)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig_publication(fig, out_path, dpi=250, bbox_inches=None)
    plt.close(fig)


def _labels_for_panel(rows: dict, clusters: np.ndarray, panel: str):
    if panel == "plain":
        return "scores", "samples", np.full(clusters.shape, "samples", dtype="U16"), False, False, False, False
    if panel == "group":
        return "model", "model", np.asarray(rows["group"]).astype(str), False, False, False, True
    if panel == "supermodel":
        return "supermodel", "supermodel", np.asarray(rows["group"]).astype(str), False, False, False, True
    if panel == "component":
        comps = np.array([MODEL_LABELS.get(str(x), str(x)) for x in rows["component"]], dtype="U32")
        return "component", "component", comps, False, False, False, True
    if panel == "cluster":
        return "clusters", "cluster", (clusters + 1).astype(str), True, True, True, True
    raise ValueError(f"Unknown PCA panel {panel!r}.")


def _plot_pca_outputs(
    scores: np.ndarray,
    explained: np.ndarray,
    rows: dict,
    clusters: np.ndarray,
    panels: list[str],
    out_prefix: Path,
    *,
    view_elev: float = 28.0,
    view_azim: float = -60.0,
):

    for panel in panels:
        suffix, legend_title, labels, is_cluster, centroid_labels, ellipses, show_legend = _labels_for_panel(rows, clusters, panel)
        out_2d = out_prefix.with_name(f"{out_prefix.name}_{suffix}_2d").with_suffix(".pdf")
        out_3d = out_prefix.with_name(f"{out_prefix.name}_{suffix}_3d").with_suffix(".pdf")
        _plot_single_pca_2d(
            scores,
            explained,
            labels,
            rows["weight"],
            out_2d,
            legend_title=legend_title,
            cluster_panel=is_cluster,
            centroid_labels=centroid_labels,
            ellipses=ellipses,
            show_legend=show_legend,
        )
        _plot_single_pca_3d(
            scores,
            explained,
            labels,
            rows["weight"],
            out_3d,
            legend_title=legend_title,
            cluster_panel=is_cluster,
            centroid_labels=centroid_labels,
            show_legend=show_legend,
            view_elev=view_elev,
            view_azim=view_azim,
        )


def _plot_pca_panels(
    scores: np.ndarray,
    explained: np.ndarray,
    panels: list[tuple[str, np.ndarray, bool]],
    out_path: Path,
):
    n_panels = len(panels)
    fig_width = 3.35 * n_panels
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, 3.15), squeeze=False)

    for ax, (legend_title, labels, is_cluster) in zip(axes.ravel(), panels):
        _scatter_panel(
            ax,
            scores,
            labels,
            explained,
            legend_title=legend_title,
            cluster_panel=is_cluster,
            centroid_labels=is_cluster,
            ellipses=is_cluster,
        )

    fig.tight_layout(pad=0.7)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig_publication(fig, out_path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def _write_scores_csv(path: Path, rows: dict, scores: np.ndarray, clusters: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "supermodel_or_model",
            "component_model",
            "cluster",
            "PC1",
            "PC2",
            "PC3",
            *FEATURE_NAMES,
        ])
        for i in range(scores.shape[0]):
            writer.writerow([
                rows["group"][i],
                MODEL_LABELS.get(str(rows["component"][i]), str(rows["component"][i])),
                int(clusters[i]) + 1,
                f"{scores[i, 0]:.10g}",
                f"{scores[i, 1]:.10g}",
                f"{scores[i, 2]:.10g}",
                *[f"{rows['X'][i, j]:.10g}" for j in range(len(FEATURE_NAMES))],
            ])


def _write_loadings_csv(path: Path, loadings: np.ndarray, explained: np.ndarray, scaling: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "feature",
            "display_label",
            "center",
            "scale",
            "clip_low",
            "clip_high",
            "PC1_loading",
            "PC2_loading",
            "PC3_loading",
        ])
        for i, name in enumerate(FEATURE_NAMES):
            writer.writerow([
                name,
                FEATURE_LABELS[name],
                f"{scaling['center'][i]:.10g}",
                f"{scaling['scale'][i]:.10g}",
                f"{scaling['lo'][i]:.10g}",
                f"{scaling['hi'][i]:.10g}",
                f"{loadings[i, 0]:.10g}",
                f"{loadings[i, 1]:.10g}",
                f"{loadings[i, 2]:.10g}",
            ])
        writer.writerow([])
        writer.writerow(["explained_variance", "", "", "", "", "", f"{explained[0]:.10g}", f"{explained[1]:.10g}", f"{explained[2]:.10g}"])


def _dominant_label(labels: np.ndarray) -> str:
    labels = np.asarray(labels).astype(str)
    if labels.size == 0:
        return ""
    vals, counts = np.unique(labels, return_counts=True)
    return str(vals[np.argmax(counts)])


def _write_cluster_summary_csv(path: Path, rows: dict, clusters: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(clusters.size)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cluster",
            "n",
            "share_pct",
            "dominant_group",
            "dominant_component",
            *[f"{name}_q05" for name in FEATURE_NAMES],
            *[f"{name}_median" for name in FEATURE_NAMES],
            *[f"{name}_q95" for name in FEATURE_NAMES],
        ])
        for c in sorted(np.unique(clusters)):
            mask = clusters == c
            Xc = rows["X"][mask]
            q05 = np.nanquantile(Xc, 0.05, axis=0)
            q50 = np.nanquantile(Xc, 0.50, axis=0)
            q95 = np.nanquantile(Xc, 0.95, axis=0)
            writer.writerow([
                int(c) + 1,
                int(np.sum(mask)),
                f"{100.0 * np.sum(mask) / max(1, n):.4g}",
                _dominant_label(rows["group"][mask]),
                MODEL_LABELS.get(_dominant_label(rows["component"][mask]), _dominant_label(rows["component"][mask])),
                *[f"{v:.10g}" for v in q05],
                *[f"{v:.10g}" for v in q50],
                *[f"{v:.10g}" for v in q95],
            ])



def _write_centroids_csv(path: Path, rows: dict, scores: np.ndarray, clusters: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cluster",
            "n",
            "weight_sum",
            "PC1_centroid",
            "PC2_centroid",
            "PC3_centroid",
            *[f"{name}_centroid" for name in FEATURE_NAMES],
        ])
        for c in sorted(np.unique(clusters)):
            mask = clusters == c
            w = np.asarray(rows["weight"])[mask]
            score_centroid = _weighted_centroid(scores[mask, :3], w)
            feature_centroid = _weighted_centroid(rows["X"][mask], w)
            writer.writerow([
                int(c) + 1,
                int(np.sum(mask)),
                f"{float(np.sum(w)):.10g}",
                *[f"{v:.10g}" for v in score_centroid],
                *[f"{v:.10g}" for v in feature_centroid],
            ])


def _print_centroids(out_prefix: Path, rows: dict, scores: np.ndarray, clusters: np.ndarray):
    print(f"[PCA centroids] {out_prefix.name}")
    for c in sorted(np.unique(clusters)):
        mask = clusters == c
        w = np.asarray(rows["weight"])[mask]
        score_centroid = _weighted_centroid(scores[mask, :3], w)
        feature_centroid = _weighted_centroid(rows["X"][mask], w)
        print(
            f"  cluster {int(c)+1}: "
            f"PC=({score_centroid[0]:.3g}, {score_centroid[1]:.3g}, {score_centroid[2]:.3g}); "
            f"logL={feature_centroid[0]:.3g}, logN_site={feature_centroid[1]:.3g}, "
            f"logN_source={feature_centroid[2]:.3g}, log_site/source={feature_centroid[3]:.3g}, "
            f"cap_fraction={feature_centroid[4]:.3g}"
        )


def _run_one_pca(
    rows: dict,
    *,
    out_prefix: Path,
    n_clusters: int,
    cluster_pcs: int,
    seed: int,
    panels: list[str],
    view_elev: float = 28.0,
    view_azim: float = -60.0,
):
    Z, scaling = _robust_standardize(rows["X"])
    scores, explained, loadings = _pca_scores(Z, n_components=3)
    if scores.shape[1] >= 3:
        scores[:, 2] *= -1.0
        loadings[:, 2] *= -1.0

    cluster_pcs = int(max(1, min(cluster_pcs, scores.shape[1])))
    clusters, _centers = _kmeans(scores[:, :cluster_pcs], n_clusters=n_clusters, rng=np.random.default_rng(seed))


    _plot_pca_outputs(
        scores,
        explained,
        rows,
        clusters,
        panels,
        out_prefix,
        view_elev=view_elev,
        view_azim=view_azim,
    )
    _write_scores_csv(out_prefix.with_name(out_prefix.name + "_scores.csv"), rows, scores, clusters)
    _write_cluster_summary_csv(out_prefix.with_name(out_prefix.name + "_cluster_summary.csv"), rows, clusters)
    _write_centroids_csv(out_prefix.with_name(out_prefix.name + "_cluster_centroids.csv"), rows, scores, clusters)
    _print_centroids(out_prefix, rows, scores, clusters)
    _write_loadings_csv(out_prefix.with_name(out_prefix.name + "_loadings.csv"), loadings, explained, scaling)


def _load_available(paths: list[Path], *, max_rows: int, lifetime_key: str, seed: int, fixed_groups: list[str] | None = None):
    row_sets = []
    rng_master = np.random.default_rng(seed)
    for i, path in enumerate(paths):
        if not path.exists():
            print(f"[WARN] PCA skipped missing file: {path}")
            continue
        fixed = fixed_groups[i] if fixed_groups is not None else None
        try:
            row_sets.append(
                _load_sample_rows(
                    path,
                    max_rows=max_rows,
                    rng=np.random.default_rng(rng_master.integers(0, 2**31 - 1)),
                    lifetime_key=lifetime_key,
                    fixed_group=fixed,
                )
            )
        except Exception as exc:
            print(f"[WARN] PCA skipped {path}: {exc}")
    if not row_sets:
        return None
    return _concat_rows(row_sets)


def make_all_basic_models_sample_pca(
    sample_dir: Path,
    out_dir: Path,
    *,
    lifetime_key: str,
    rows_per_model: int,
    n_clusters: int,
    cluster_pcs: int,
    seed: int,
    view_elev: float = 28.0,
    view_azim: float = -60.0,
):
    paths = [sample_dir / f"{model_id}.npz" for model_id in MODEL_NAMES]
    groups = [MODEL_LABELS.get(model_id, model_id) for model_id in MODEL_NAMES]
    rows = _load_available(paths, max_rows=rows_per_model, lifetime_key=lifetime_key, seed=seed, fixed_groups=groups)
    if rows is None:
        print("[WARN] sample-level basic PCA skipped: no valid files")
        return
    _run_one_pca(
        rows,
        out_prefix=out_dir / "basic_models" / "all_basic_models_sample_pca",
        n_clusters=n_clusters,
        cluster_pcs=cluster_pcs,
        seed=seed + 111,
        panels=["group", "cluster"],
        view_elev=view_elev,
        view_azim=view_azim,
    )



def make_each_basic_model_sample_pca(
    sample_dir: Path,
    out_dir: Path,
    *,
    lifetime_key: str,
    rows_per_model: int,
    n_clusters: int,
    cluster_pcs: int,
    seed: int,
    view_elev: float = 28.0,
    view_azim: float = -60.0,
):

    for i, model_id in enumerate(MODEL_NAMES):
        path = sample_dir / f"{model_id}.npz"
        label = MODEL_LABELS.get(model_id, model_id)
        rows = _load_available([path], max_rows=rows_per_model, lifetime_key=lifetime_key, seed=seed + 1000 * i, fixed_groups=[label])
        if rows is None:
            print(f"[WARN] sample-level PCA skipped for {model_id}: no valid file")
            continue
        _run_one_pca(
            rows,
            out_prefix=out_dir / "basic_models" / f"{model_id}_sample_pca",
            n_clusters=n_clusters,
            cluster_pcs=cluster_pcs,
            seed=seed + 444 + 1000 * i,
            panels=["plain", "cluster"],
            view_elev=view_elev,
            view_azim=view_azim,
        )


def make_all_supermodels_sample_pca(
    super_sample_dir: Path,
    out_dir: Path,
    *,
    lifetime_key: str,
    rows_per_supermodel: int,
    n_clusters: int,
    cluster_pcs: int,
    seed: int,
    view_elev: float = 28.0,
    view_azim: float = -60.0,
):
    paths = [super_sample_dir / f"{super_id}.npz" for super_id in SUPERMODEL_NAMES]
    rows = _load_available(paths, max_rows=rows_per_supermodel, lifetime_key=lifetime_key, seed=seed, fixed_groups=SUPERMODEL_NAMES)
    if rows is None:
        print("[WARN] sample-level supermodel PCA skipped: no valid files")
        return
    _run_one_pca(
        rows,
        out_prefix=out_dir / "supermodels" / "all_supermodels_sample_pca",
        n_clusters=n_clusters,
        cluster_pcs=cluster_pcs,
        seed=seed + 222,
        panels=["supermodel", "component", "cluster"],
        view_elev=view_elev,
        view_azim=view_azim,
    )


def make_each_supermodel_sample_pca(
    super_sample_dir: Path,
    out_dir: Path,
    *,
    lifetime_key: str,
    rows_per_supermodel: int,
    n_clusters: int,
    cluster_pcs: int,
    seed: int,
    view_elev: float = 28.0,
    view_azim: float = -60.0,
):
    for i, super_id in enumerate(SUPERMODEL_NAMES):
        path = super_sample_dir / f"{super_id}.npz"
        rows = _load_available([path], max_rows=rows_per_supermodel, lifetime_key=lifetime_key, seed=seed + 1000 * i, fixed_groups=[super_id])
        if rows is None:
            print(f"[WARN] sample-level PCA skipped for {super_id}: no valid file")
            continue
        _run_one_pca(
            rows,
            out_prefix=out_dir / "supermodels" / f"{super_id}_sample_pca",
            n_clusters=n_clusters,
            cluster_pcs=cluster_pcs,
            seed=seed + 333 + 1000 * i,
            panels=["component", "cluster"],
            view_elev=view_elev,
            view_azim=view_azim,
        )


def run_pca_analysis(
    outdir: Path,
    pca_mode: str = "sample",
    lifetime_key: str = "L_active",
    logL_min: float = 0.0,
    logL_max: float = 8.0,
    logN_min: float = 0.0,
    logN_max: float = 12.0,
    bins_L: int = 56,
    bins_N: int = 56,
    chunk_size: int = 2500,
    basic_chunks: int = 60,
    super_chunks: int = 120,
    basic_clusters: int = 5,
    super_clusters: int = 6,
    seed: int = 12345,
    rows_per_model: int = 6_000,
    rows_per_supermodel: int = 6_000,
    cluster_pcs: int = 3,
    view_elev: float = 28.0,
    view_azim: float = -60.0,
):

    sample_dir = outdir / "samples"
    super_sample_dir = outdir / "super_samples"
    pca_dir = outdir / "pca"
    pca_dir.mkdir(parents=True, exist_ok=True)

    if pca_mode.upper() in {"L", "LN"}:
        print("[INFO] Histogram PCA modes L/LN are retired. Running sample-level PCA instead.")

    make_all_basic_models_sample_pca(
        sample_dir=sample_dir,
        out_dir=pca_dir,
        lifetime_key=lifetime_key,
        rows_per_model=rows_per_model,
        n_clusters=basic_clusters,
        cluster_pcs=cluster_pcs,
        seed=seed + 700,
        view_elev=view_elev,
        view_azim=view_azim,
    )

    make_each_basic_model_sample_pca(
        sample_dir=sample_dir,
        out_dir=pca_dir,
        lifetime_key=lifetime_key,
        rows_per_model=rows_per_model,
        n_clusters=basic_clusters,
        cluster_pcs=cluster_pcs,
        seed=seed + 750,
        view_elev=view_elev,
        view_azim=view_azim,
    )

    make_all_supermodels_sample_pca(
        super_sample_dir=super_sample_dir,
        out_dir=pca_dir,
        lifetime_key=lifetime_key,
        rows_per_supermodel=rows_per_supermodel,
        n_clusters=super_clusters,
        cluster_pcs=cluster_pcs,
        seed=seed + 800,
        view_elev=view_elev,
        view_azim=view_azim,
    )

    make_each_supermodel_sample_pca(
        super_sample_dir=super_sample_dir,
        out_dir=pca_dir,
        lifetime_key=lifetime_key,
        rows_per_supermodel=rows_per_supermodel,
        n_clusters=super_clusters,
        cluster_pcs=cluster_pcs,
        seed=seed + 900,
        view_elev=view_elev,
        view_azim=view_azim,
    )

    print(f"[OK] sample-level PCA analysis finished: {pca_dir}")
