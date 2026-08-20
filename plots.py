from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from figure_style import configure_matplotlib_for_latex, savefig_publication
configure_matplotlib_for_latex()

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None


MODEL_DISPLAY_NAMES = {
    "model1": "Model I",
    "model2": "Model II",
    "model3a": "Model IIIA",
    "model3b": "Model IIIB-M",
    "model4": "Model IV",
    "SM1": "SM1",
    "SM2": "SM2",
    "SM3": "SM3",
    "SM4": "SM4",
    "SM5": "SM5",
}

BASIC_MODEL_IDS = {"model1", "model2", "model3a", "model3b", "model4"}
DIST_CODE_LABELS = {
    0: "lognormal",
    1: "loglinear",
    2: "gauss",
}


def finite_positive(x):
    return np.isfinite(x) & (x > 0.0)


def infer_distribution_label(data) -> str | None:
    if "sampling_distribution" in data.files:
        arr = np.asarray(data["sampling_distribution"])
        if arr.size >= 1:
            return str(arr.reshape(-1)[0])


    if "super_dist" in data.files:
        arr = np.asarray(data["super_dist"])
        if arr.size >= 1:
            return str(arr.reshape(-1)[0])

    codes: set[int] = set()
    for key in data.files:
        if not key.startswith("dist_"):
            continue
        values = np.asarray(data[key]).reshape(-1)
        values = values[np.isfinite(values)] if np.issubdtype(values.dtype, np.number) else values
        for value in values:
            code = int(value)
            if code >= 0:
                codes.add(code)

    if not codes:
        return None
    if len(codes) == 1:
        return DIST_CODE_LABELS.get(next(iter(codes)), None)
    return "mixed"


def display_title(model_id: str, distribution_label: str | None = None) -> str:
    model_id = str(model_id)
    base = MODEL_DISPLAY_NAMES.get(model_id, model_id)
    if model_id in BASIC_MODEL_IDS and distribution_label:
        return f"{base} distributed by {distribution_label}"
    return base




def _smooth2d_box_kernel(Z, passes=2):
    if passes <= 0:
        return Z

    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float)
    kernel /= kernel.sum()

    out = Z.copy()
    for _ in range(int(np.ceil(passes))):
        pad = np.pad(out, 1, mode="edge")
        new = np.zeros_like(out)
        for i in range(3):
            for j in range(3):
                new += kernel[i, j] * pad[i:i + out.shape[0], j:j + out.shape[1]]
        out = new

    return out


def smooth2d(Z, passes=3.0):
    strength = float(passes)
    if strength <= 0:
        return Z

    if gaussian_filter is not None:
        return gaussian_filter(Z, sigma=strength, mode="nearest")

    return _smooth2d_box_kernel(Z, passes=strength)




def gaussian_kernel_1d(sigma_bins=2.0):
    if sigma_bins <= 0:
        return np.array([1.0])

    radius = int(np.ceil(4.0 * sigma_bins))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= np.sum(kernel)
    return kernel


def smooth1d(y, sigma_bins=2.0):
    y = np.asarray(y, dtype=float)
    if sigma_bins <= 0:
        return y.copy()

    kernel = gaussian_kernel_1d(sigma_bins)
    radius = len(kernel) // 2
    padded = np.pad(y, radius, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def weighted_range(x, qlo=0.005, qhi=0.995):
    lo, hi = np.quantile(x, [qlo, qhi])
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def density1d(x, weight, bins=120, x_range=None, smooth_sigma=2.0):
    x = np.asarray(x, dtype=float)
    weight = np.asarray(weight, dtype=float)

    ok = np.isfinite(x) & np.isfinite(weight) & (weight >= 0.0)
    x = x[ok]
    weight = weight[ok]

    if x.size == 0:
        raise ValueError("No valid samples for 1D density.")
    if np.sum(weight) <= 0.0:
        weight = np.ones_like(x)

    if x_range is None:
        x_range = weighted_range(x)

    counts, edges = np.histogram(x, bins=bins, range=x_range, weights=weight, density=False)
    counts = smooth1d(counts, sigma_bins=smooth_sigma)

    centers = 0.5 * (edges[:-1] + edges[1:])
    dx = edges[1] - edges[0]
    norm = np.sum(counts) * dx

    if norm <= 0.0 or not np.isfinite(norm):
        pdf = np.zeros_like(centers)
    else:
        pdf = counts / norm

    return centers, pdf


def survival_from_pdf(x, pdf):
    x = np.asarray(x, dtype=float)
    pdf = np.asarray(pdf, dtype=float)

    if x.size != pdf.size:
        raise ValueError("x and pdf must have equal length.")
    if x.size < 2:
        raise ValueError("Need at least two points for survival.")

    dx = x[1] - x[0]
    mass = pdf * dx
    survival = np.cumsum(mass[::-1])[::-1]
    if survival[0] > 0:
        survival /= survival[0]
    return survival



def density2d(
    logL,
    logN,
    weight,
    bins=90,
    smooth_passes=3.0,
    logN_floor=0.0,
    logL_range=None,
    logN_range=None,
):

    if logL_range is None:
        xr = weighted_range(logL)
    else:
        xr = tuple(map(float, logL_range))

    if logN_range is None:
        yr_lo, yr_hi = weighted_range(logN)
    else:
        yr_lo, yr_hi = map(float, logN_range)

    if logN_floor is not None:
        yr_lo = max(float(logN_floor), float(yr_lo))
        yr_hi = max(float(yr_hi), yr_lo + 1.0e-6)

    yr = (float(yr_lo), float(yr_hi))

    H, xedges, yedges = np.histogram2d(
        logL,
        logN,
        bins=bins,
        range=[xr, yr],
        weights=weight,
        density=False,
    )

    H = smooth2d(H, passes=smooth_passes)

    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]
    norm = np.sum(H) * dx * dy
    if norm > 0:
        H = H / norm

    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(xc, yc, indexing="ij")

    return X, Y, H, xedges, yedges




def _weighted_center_2d(x, y, w):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w >= 0.0)
    if not np.any(ok):
        return float(np.mean(x)), float(np.mean(y))
    if float(np.sum(w[ok])) <= 0.0:
        return float(np.mean(x[ok])), float(np.mean(y[ok]))
    return float(np.average(x[ok], weights=w[ok])), float(np.average(y[ok], weights=w[ok]))


def _surface_height_at(X, Y, Z, x0: float, y0: float) -> float:
    ix = int(np.argmin(np.abs(X[:, 0] - x0)))
    iy = int(np.argmin(np.abs(Y[0, :] - y0)))
    return float(Z[ix, iy])


def _append_surface_centroid_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8") as f:
        if not exists:
            f.write("name,logL_centroid,logN_centroid,L_centroid,N_centroid\n")
        f.write(
            f"{row['name']},{row['logL']:.10g},{row['logN']:.10g},"
            f"{10.0 ** row['logL']:.10g},{10.0 ** row['logN']:.10g}\n"
        )


def plot_surface(
    X,
    Y,
    Z,
    path,
    title,
    center_point=None,
    surface_elev: float = 28.0,
    surface_azim: float = -60.0,
    surface_cmap: str = "viridis",
):
    fig = plt.figure(figsize=(8.2, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap=surface_cmap, linewidth=0, antialiased=True)

    ax.set_xlabel(r"$\log(L)$")
    ax.set_ylabel(r"$\log(N)$")
    ax.set_zlabel(r"density")
    ax.set_title(title)
    ax.set_ylim(0.0, float(np.nanmax(Y)))
    ax.margins(x=0.0, y=0.0, z=0.0)
    ax.view_init(elev=float(surface_elev), azim=float(surface_azim))
    fig.tight_layout()
    savefig_publication(fig, path, dpi=300)
    plt.close(fig)


def plot_heatmap(Z, xedges, yedges, path, title, center_point=None):
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    im = ax.imshow(
        Z.T,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
        cmap="viridis",
    )
    ax.set_xlabel(r"$\log(L)$")
    ax.set_ylabel(r"$\log(N)$")
    ax.set_title(title)

    if center_point is not None:
        xc, yc = map(float, center_point)
        if np.isfinite(xc) and np.isfinite(yc):
            ax.scatter(
                [xc],
                [yc],
                s=18,
                marker="o",
                c="red",
                edgecolors="none",
                linewidths=0.0,
                zorder=20,
            )


    ax.set_ylim(0.0, yedges[-1])
    ax.margins(x=0.0, y=0.0)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("density")
    fig.tight_layout()
    savefig_publication(fig, path, dpi=300)
    plt.close(fig)


def plot_tripanel(
    logL,
    logN,
    weight,
    path,
    title,
    bins=120,
    smooth_sigma=2.0,
    logL_range=None,
    logN_range=None,
):
    
    if logL_range is None:
        l_range = weighted_range(logL)
        l_range = (0.0, max(l_range[1], 1.0e-6))
    else:
        l_range = tuple(map(float, logL_range))

    if logN_range is None:
        n_range = weighted_range(logN)
        n_range = (0.0, max(n_range[1], 1.0e-6))
    else:
        n_range = tuple(map(float, logN_range))

    l_range = (max(0.0, l_range[0]), max(l_range[1], max(0.0, l_range[0]) + 1.0e-6))
    n_range = (max(0.0, n_range[0]), max(n_range[1], max(0.0, n_range[0]) + 1.0e-6))

    xL, pdfL = density1d(logL, weight, bins=bins, x_range=l_range, smooth_sigma=smooth_sigma)
    xN, pdfN = density1d(logN, weight, bins=bins, x_range=n_range, smooth_sigma=smooth_sigma)
    surv = survival_from_pdf(xL, pdfL)

    left_L = l_range[0]
    left_N = n_range[0]

    xL_pdf = np.concatenate(([left_L], xL))
    pdfL_plot = np.concatenate(([pdfL[0]], pdfL))
    xL_surv = np.concatenate(([left_L], xL))
    surv_plot = np.concatenate(([1.0], surv))
    xN_pdf = np.concatenate(([left_N], xN))
    pdfN_plot = np.concatenate(([pdfN[0]], pdfN))

    fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.2))

    ax[0].plot(xL_pdf, pdfL_plot, linewidth=2.0)
    ax[0].set_xlabel(r"$\log(L)$")
    ax[0].set_ylabel("density")
    ax[0].set_title("Density by L")
    ax[0].grid(True, alpha=0.25)

    ax[1].plot(xL_surv, surv_plot, linewidth=2.0)
    ax[1].set_xlabel(r"$\log(L)$")
    ax[1].set_ylabel("survival")
    ax[1].set_ylim(-0.02, 1.02)
    ax[1].set_title("Survival by L")
    ax[1].grid(True, alpha=0.25)

    ax[2].plot(xN_pdf, pdfN_plot, linewidth=2.0)
    ax[2].set_xlabel(r"$\log(N)$")
    ax[2].set_ylabel("density")
    ax[2].set_title("Density by N")
    ax[2].grid(True, alpha=0.25)

    ax[0].set_xlim(l_range[0], l_range[1])
    ax[1].set_xlim(l_range[0], l_range[1])
    ax[2].set_xlim(n_range[0], n_range[1])

    fig.suptitle(title)
    fig.tight_layout()
    savefig_publication(fig, path, dpi=300)
    plt.close(fig)


def make_model_plots(
    npz_path: Path,
    outdir: Path,
    bins=90,
    source=False,
    smooth1d_sigma=2.0,
    smooth2d_passes=3.0,
    lifetime_key="L_active",
    logL_range=None,
    logN_range=None,
    surface_elev: float = 28.0,
    surface_azim: float = -60.0,
    surface_cmap: str = "viridis",
    super_surface_cmap: str | None = None,
    **_unused_plot_kwargs,
):
    data = np.load(npz_path, allow_pickle=True)
    model_id = str(data["model_id"])

    if lifetime_key not in data.files:
        raise KeyError(f"{npz_path} does not contain {lifetime_key!r}.")

    L = data[lifetime_key]
    N = data["N_source"] if source else data["N_active"]
    weight = data["weight"]

    ok = finite_positive(L) & finite_positive(N) & np.isfinite(weight) & (weight >= 0.0)
    L, N, weight = L[ok], N[ok], weight[ok]
    if np.sum(weight) <= 0.0:
        weight = np.ones_like(L)

    logL = np.log10(L)
    logN = np.log10(N)

    label = "N_source" if source else "N_active"
    if lifetime_key == "L_active":
        name = f"{model_id}_{label}"
    else:
        name = f"{model_id}_{label}_{lifetime_key}"

    title = display_title(model_id, infer_distribution_label(data))
    if source:
        title = f"{title} -- source population"
    elif str(model_id).startswith("SM"):
        title = f"{title} -- active sites"

    X, Y, Z, xedges, yedges = density2d(
        logL,
        logN,
        weight,
        bins=bins,
        smooth_passes=smooth2d_passes,
        logN_floor=0.0,
        logL_range=logL_range,
        logN_range=logN_range,
    )

    center_point = _weighted_center_2d(logL, logN, weight)
    print(
        f"[surface centroid] {name}: "
        f"logL={center_point[0]:.4g}, logN={center_point[1]:.4g}; "
        f"L={10.0 ** center_point[0]:.4g}, N={10.0 ** center_point[1]:.4g}"
    )
    _append_surface_centroid_csv(
        outdir / "surface_centroids.csv",
        {"name": name, "logL": center_point[0], "logN": center_point[1]},
    )
    surface_cmap_effective = surface_cmap
    if str(model_id).startswith("SM") and super_surface_cmap is not None:
        surface_cmap_effective = super_surface_cmap

    plot_surface(
        X,
        Y,
        Z,
        outdir / f"{name}_surface.png",
        title,
        center_point=center_point,
        surface_elev=surface_elev,
        surface_azim=surface_azim,
        surface_cmap=surface_cmap_effective,
    )
    plot_heatmap(Z, xedges, yedges, outdir / f"{name}_heatmap.png", title, center_point=center_point)
    plot_tripanel(
        logL,
        logN,
        weight,
        outdir / f"{name}_tripanel.png",
        title,
        bins=max(bins, 120),
        smooth_sigma=smooth1d_sigma,
        logL_range=logL_range,
        logN_range=logN_range,
    )
