from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from figure_style import configure_matplotlib_for_latex, savefig_publication
configure_matplotlib_for_latex()

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def resample_until_inside(
    values: np.ndarray,
    draw,
    lo: float,
    hi: float,
    max_iter: int = 100,
) -> np.ndarray:
    mask = (values < lo) | (values > hi)

    it = 0
    while np.any(mask) and it < max_iter:
        values[mask] = draw(int(np.sum(mask)))
        mask = (values < lo) | (values > hi)
        it += 1

    return np.clip(values, lo, hi)


def sample_lognormal(
    lo: float,
    peak: float,
    hi: float,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    a, c, b = np.log([lo, peak, hi])
    sigma = max(abs(c - a), abs(b - c)) / 3.0

    def draw(k: int) -> np.ndarray:
        return rng.normal(c, sigma, size=k)

    y = draw(n)
    y = resample_until_inside(y, draw, a, b)
    return np.exp(y)


def sample_loglinear(
    lo: float,
    peak: float,
    hi: float,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    a, c, b = np.log10([lo, peak, hi])
    y = rng.triangular(a, c, b, size=n)
    return 10.0 ** y


def sample_gauss(
    lo: float,
    peak: float,
    hi: float,
    n: int,
    rng: np.random.Generator,
    sigma: float | None = None,
) -> np.ndarray:
    if sigma is None:
        sigma = peak

    def draw(k: int) -> np.ndarray:
        return rng.normal(peak, sigma, size=k)

    x = draw(n)
    return resample_until_inside(x, draw, lo, hi)


def smoothed_density_histogram(
    x: np.ndarray,
    bins: int,
    x_range: tuple[float, float],
    smooth_sigma: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(x, bins=bins, range=x_range, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    
    counts_smooth = gaussian_filter1d(counts, sigma=smooth_sigma, mode="nearest")

    return centers, counts_smooth


def run_distribution_demo(outdir: str | Path = "outputs/distributions") -> None:
    rng = np.random.default_rng(12345)

    lo = 1.0
    peak = 3.16
    hi = 1000.0
    n = 400_000

    samples = {
        "lognormal": sample_lognormal(lo, peak, hi, n, rng),
        "gauss": sample_gauss(lo, peak, hi, n, rng),
        "loglinear": sample_loglinear(lo, peak, hi, n, rng),
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))


    log_bins = 140
    log_range = (np.log10(lo), np.log10(hi))

    for label, x in samples.items():
        logx = np.log10(x)
        centers, density = smoothed_density_histogram(
            logx,
            bins=log_bins,
            x_range=log_range,
            smooth_sigma=2.0,
        )
        axes[0].plot(centers, density, label=label, linewidth=2.0)

    axes[0].set_xlabel(r"$\log(x)$")
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=True)

    x_bins = 180
    x_plot_max = 32.0
    x_range = (lo, x_plot_max)

    for label, x in samples.items():
        centers, density = smoothed_density_histogram(
            x,
            bins=x_bins,
            x_range=x_range,
            smooth_sigma=2.0,
        )
        axes[1].plot(centers, density, label=label, linewidth=2.0)

    axes[1].set_xlabel(r"$x$")
    axes[1].set_ylabel("")
    axes[1].legend(frameon=True)

    fig.tight_layout()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    savefig_publication(
        fig,
        outdir / "distribution_comparison_smoothed.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"[OK] wrote: {outdir / 'distribution_comparison_smoothed.pdf'}")
    print(f"[OK] wrote: {outdir / 'distribution_comparison_smoothed.pdf'}")


def main() -> None:
    run_distribution_demo()


if __name__ == "__main__":
    main()