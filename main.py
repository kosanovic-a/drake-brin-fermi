from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from triads import TRIADS, MODEL_NAMES
from sampler import sample_params, DISTRIBUTIONS
from models import MODELS
from plots import make_model_plots
from supermodels import generate_all_supermodels, SUPERMODEL_NAMES
from pca_analysis import run_pca_analysis
from result_summary import run_result_summary


def clean(outdir: Path):
    if outdir.exists():
        shutil.rmtree(outdir)
    (outdir / "samples").mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    (outdir / "super_samples").mkdir(parents=True, exist_ok=True)
    (outdir / "super_figures").mkdir(parents=True, exist_ok=True)
    (outdir / "pca").mkdir(parents=True, exist_ok=True)
    (outdir / "summary").mkdir(parents=True, exist_ok=True)
    print(f"[OK] clean: {outdir}")


def generate(outdir: Path, dist: str, samples: int, seed: int):
    sample_dir = outdir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    for model_id in MODEL_NAMES:
        p, dist_codes = sample_params(TRIADS[model_id], samples, dist, rng, return_dist_codes=True)
        out = MODELS[model_id](p)
        out.update(p)
        out.update(dist_codes)
        path = sample_dir / f"{model_id}.npz"
        np.savez_compressed(path, **out)
        print(f"[OK] generated: {path}")


def plot(
    outdir: Path,
    bins: int,
    smooth1d: float,
    smooth2d: float,
    surface_elev: float,
    surface_azim: float,
):
    sample_dir = outdir / "samples"
    fig_dir = outdir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for model_id in MODEL_NAMES:
        path = sample_dir / f"{model_id}.npz"
        make_model_plots(
            path,
            fig_dir,
            bins=bins,
            source=False,
            smooth1d_sigma=smooth1d,
            smooth2d_passes=smooth2d,
            surface_elev=surface_elev,
            surface_azim=surface_azim,
        )
        if model_id in {"model3a", "model3b"}:
            make_model_plots(
                path,
                fig_dir,
                bins=bins,
                source=True,
                smooth1d_sigma=smooth1d,
                smooth2d_passes=smooth2d,
                surface_elev=surface_elev,
                surface_azim=surface_azim,
            )
            make_model_plots(
                path,
                fig_dir,
                bins=bins,
                source=False,
                smooth1d_sigma=smooth1d,
                smooth2d_passes=smooth2d,
                lifetime_key="L_surv",
                surface_elev=surface_elev,
                surface_azim=surface_azim,
            )
            make_model_plots(
                path,
                fig_dir,
                bins=bins,
                source=True,
                smooth1d_sigma=smooth1d,
                smooth2d_passes=smooth2d,
                lifetime_key="L_surv",
                surface_elev=surface_elev,
                surface_azim=surface_azim,
            )

        print(f"[OK] plotted: {model_id}")

def supermodels(
    outdir: Path,
    bins: int,
    smooth1d: float,
    smooth2d: float,
    samples: int,
    super_dist: str,
    seed: int,
    super_logL_min: float,
    super_logL_max: float,
    super_logN_active_min: float,
    super_logN_active_max: float,
    super_logN_source_min: float,
    super_logN_source_max: float,
    super_surface_cmap: str,
    surface_elev: float,
    surface_azim: float,
):
    super_sample_dir = outdir / "super_samples"
    super_fig_dir = outdir / "super_figures"

    super_sample_dir.mkdir(parents=True, exist_ok=True)
    super_fig_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed + 1_000_003)
    generate_all_supermodels(
        outdir=super_sample_dir,
        samples_per_component=samples,
        super_dist=super_dist,
        rng=rng,
    )

    logL_range = (super_logL_min, super_logL_max)
    logN_active_range = (super_logN_active_min, super_logN_active_max)
    logN_source_range = (super_logN_source_min, super_logN_source_max)

    for super_id in SUPERMODEL_NAMES:
        path = super_sample_dir / f"{super_id}.npz"

        make_model_plots(
            path,
            super_fig_dir,
            bins=bins,
            source=False,
            smooth1d_sigma=smooth1d,
            smooth2d_passes=smooth2d,
            surface_elev=surface_elev,
            surface_azim=surface_azim,
            logL_range=logL_range,
            logN_range=logN_active_range,
            surface_cmap=super_surface_cmap,
        )


        make_model_plots(
            path,
            super_fig_dir,
            bins=bins,
            source=True,
            smooth1d_sigma=smooth1d,
            smooth2d_passes=smooth2d,
            surface_elev=surface_elev,
            surface_azim=surface_azim,
            logL_range=logL_range,
            logN_range=logN_source_range,
            surface_cmap=super_surface_cmap,
        )
        print(f"[OK] supermodel plotted: {super_id}")

    print(f"[OK] supermodel summary: {super_sample_dir / 'supermodel_summary.csv'}")


def pca(
    outdir: Path,
    pca_mode: str,
    pca_lifetime: str,
    pca_bins_l: int,
    pca_bins_n: int,
    pca_chunk_size: int,
    pca_basic_chunks: int,
    pca_super_chunks: int,
    pca_basic_clusters: int,
    pca_super_clusters: int,
    pca_logL_min: float,
    pca_logL_max: float,
    pca_logN_min: float,
    pca_logN_max: float,
    pca_rows_per_model: int,
    pca_rows_per_supermodel: int,
    pca_cluster_pcs: int,
    pca_view_elev: float,
    pca_view_azim: float,
    seed: int,
):
    run_pca_analysis(
        outdir=outdir,
        pca_mode=pca_mode,
        lifetime_key=pca_lifetime,
        logL_min=pca_logL_min,
        logL_max=pca_logL_max,
        logN_min=pca_logN_min,
        logN_max=pca_logN_max,
        bins_L=pca_bins_l,
        bins_N=pca_bins_n,
        chunk_size=pca_chunk_size,
        basic_chunks=pca_basic_chunks,
        super_chunks=pca_super_chunks,
        basic_clusters=pca_basic_clusters,
        super_clusters=pca_super_clusters,
        rows_per_model=pca_rows_per_model,
        rows_per_supermodel=pca_rows_per_supermodel,
        cluster_pcs=pca_cluster_pcs,
        view_elev=pca_view_elev,
        view_azim=pca_view_azim,
        seed=seed,
    )


def remove_intermediate_npz(outdir: Path):
    removed = 0
    for sub in ("samples", "super_samples"):
        d = outdir / sub
        if not d.exists():
            continue
        for path in d.glob("*.npz"):
            path.unlink()
            removed += 1
    if removed:
        print(f"[OK] removed intermediate .npz files: {removed}")


def parse_args():
    parser = argparse.ArgumentParser(description="CIVILIZATIONS basic and fully-random supermodel pipeline")
    parser.add_argument("actions", nargs="+", choices=["clean", "generate", "plot", "super", "pca", "summary", "importance", "distributions"])
    parser.add_argument("--out", default="outputs", help="Output directory")
    parser.add_argument("--dist", default="lognormal", choices=DISTRIBUTIONS)
    parser.add_argument(
        "--super-dist",
        default="mixed",
        choices=DISTRIBUTIONS,
        help="Distribution mode for supermodels. Default 'mixed' randomizes the distribution for every parameter/sample.",
    )
    parser.add_argument("--samples", type=int, default=120_000)
    parser.add_argument("--bins", type=int, default=90)
    parser.add_argument("--smooth1d", type=float, default=2.0, help="Tripanel smoothing width in bins; use 0 to disable")
    parser.add_argument("--smooth2d", type=float, default=3.0, help="Surface/heatmap Gaussian smoothing sigma in bins; use 0 to disable")
    parser.add_argument("--seed", type=int, default=12345)

    parser.add_argument("--super-logL-min", type=float, default=0.0, help="Fixed lower log(L) bound for supermodel comparison plots")
    parser.add_argument("--super-logL-max", type=float, default=8.0, help="Fixed upper log(L) bound for supermodel comparison plots")
    parser.add_argument("--super-logN-active-min", type=float, default=0.0, help="Fixed lower log(N_site) bound for supermodel comparison plots")
    parser.add_argument("--super-logN-active-max", type=float, default=12.0, help="Fixed upper log(N_site) bound for supermodel comparison plots")
    parser.add_argument("--super-logN-source-min", type=float, default=0.0, help="Fixed lower log(N_source) bound for supermodel comparison plots")
    parser.add_argument("--super-logN-source-max", type=float, default=10.0, help="Fixed upper log(N_source) bound for supermodel comparison plots")
    parser.add_argument("--super-surface-cmap", default="turbo", help="Matplotlib colormap used for supermodel 3D surface plots; default is a full-spectrum turbo map")
    parser.add_argument("--surface-elev", type=float, default=28.0, help="Elevation angle for all 3D surface plots; Matplotlib ax.view_init(elev=...).")
    parser.add_argument("--surface-azim", type=float, default=-60.0, help="Azimuth angle for all 3D surface plots; Matplotlib ax.view_init(azim=...).")

    parser.add_argument("--pca-mode", default="sample", choices=["sample", "L", "LN"], help="PCA mode. The current implementation uses sample-level PCA; L/LN are accepted for compatibility.")
    parser.add_argument("--pca-lifetime", default="L_active", choices=["L_active", "L_surv"], help="Lifetime field used for PCA")
    parser.add_argument("--pca-bins-l", type=int, default=56, help="Legacy histogram-PCA option retained for compatibility; ignored by sample-level PCA")
    parser.add_argument("--pca-bins-n", type=int, default=56, help="Legacy histogram-PCA option retained for compatibility; ignored by sample-level PCA")
    parser.add_argument("--pca-chunk-size", type=int, default=2500, help="Legacy histogram-PCA option retained for compatibility; ignored by sample-level PCA")
    parser.add_argument("--pca-basic-chunks", type=int, default=60, help="Legacy histogram-PCA option retained for compatibility; ignored by sample-level PCA")
    parser.add_argument("--pca-super-chunks", type=int, default=120, help="Legacy histogram-PCA option retained for compatibility; ignored by sample-level PCA")
    parser.add_argument("--pca-basic-clusters", type=int, default=5, help="KMeans clusters for basic-model sample PCA")
    parser.add_argument("--pca-super-clusters", type=int, default=7, help="KMeans clusters for supermodel sample PCA")
    parser.add_argument("--pca-rows-per-model", type=int, default=6000, help="Weighted Monte Carlo rows sampled from each basic-model npz for PCA")
    parser.add_argument("--pca-rows-per-supermodel", type=int, default=6000, help="Weighted Monte Carlo rows sampled from each supermodel npz for PCA")
    parser.add_argument("--pca-cluster-pcs", type=int, default=3, help="Number of leading PCs used for KMeans clustering")
    parser.add_argument("--pca-view-elev", type=float, default=28.0, help="Elevation angle for 3D PCA scatter plots; analogous to --surface-elev.")
    parser.add_argument("--pca-view-azim", type=float, default=-60.0, help="Azimuth angle for 3D PCA scatter plots; analogous to --surface-azim.")
    parser.add_argument("--pca-logL-min", type=float, default=0.0)
    parser.add_argument("--pca-logL-max", type=float, default=8.0)
    parser.add_argument("--pca-logN-min", type=float, default=0.0)
    parser.add_argument("--pca-logN-max", type=float, default=12.0)

    parser.add_argument("--delete-npz", action="store_true", help="Delete generated .npz sample archives after the requested actions finish. By default, .npz files are preserved for reproducibility, PCA, summaries, and repository inspection.")


    parser.add_argument("--importance-n-basic", type=int, default=80_000)
    parser.add_argument("--importance-n-super-total", type=int, default=120_000)
    parser.add_argument("--importance-rf-trees", type=int, default=300)
    parser.add_argument("--importance-perm-repeats", type=int, default=8)
    parser.add_argument("--importance-max-perm-rows", type=int, default=25_000)
    parser.add_argument("--importance-impurity", action="store_true", help="Use RF impurity importance instead of permutation importance")
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.out)

    for action in args.actions:
        if action == "clean":
            clean(outdir)
        elif action == "generate":
            generate(outdir, args.dist, args.samples, args.seed)
        elif action == "plot":
            plot(outdir, args.bins, args.smooth1d, args.smooth2d, args.surface_elev, args.surface_azim)
        elif action == "super":
            supermodels(
                outdir=outdir,
                bins=args.bins,
                smooth1d=args.smooth1d,
                smooth2d=args.smooth2d,
                samples=args.samples,
                super_dist=args.super_dist,
                seed=args.seed,
                super_logL_min=args.super_logL_min,
                super_logL_max=args.super_logL_max,
                super_logN_active_min=args.super_logN_active_min,
                super_logN_active_max=args.super_logN_active_max,
                super_logN_source_min=args.super_logN_source_min,
                super_logN_source_max=args.super_logN_source_max,
                super_surface_cmap=args.super_surface_cmap,
                surface_elev=args.surface_elev,
                surface_azim=args.surface_azim,
            )
        elif action == "pca":
            pca(
                outdir=outdir,
                pca_mode=args.pca_mode,
                pca_lifetime=args.pca_lifetime,
                pca_bins_l=args.pca_bins_l,
                pca_bins_n=args.pca_bins_n,
                pca_chunk_size=args.pca_chunk_size,
                pca_basic_chunks=args.pca_basic_chunks,
                pca_super_chunks=args.pca_super_chunks,
                pca_basic_clusters=args.pca_basic_clusters,
                pca_super_clusters=args.pca_super_clusters,
                pca_logL_min=args.pca_logL_min,
                pca_logL_max=args.pca_logL_max,
                pca_logN_min=args.pca_logN_min,
                pca_logN_max=args.pca_logN_max,
                pca_rows_per_model=args.pca_rows_per_model,
                pca_rows_per_supermodel=args.pca_rows_per_supermodel,
                pca_cluster_pcs=args.pca_cluster_pcs,
                pca_view_elev=args.pca_view_elev,
                pca_view_azim=args.pca_view_azim,
                seed=args.seed,
            )
        elif action == "summary":
            run_result_summary(outdir)
        elif action == "importance":
            from importance_grid import run_importance_grid

            run_importance_grid(
                out=outdir,
                dist=args.dist,
                super_dist=args.super_dist,
                n_basic=args.importance_n_basic,
                n_super_total=args.importance_n_super_total,
                lifetime_key="L_active",
                seed=args.seed,
                rf_trees=args.importance_rf_trees,
                use_permutation=not args.importance_impurity,
                perm_repeats=args.importance_perm_repeats,
                max_perm_rows=args.importance_max_perm_rows,
            )
        elif action == "distributions":
            from distributions import run_distribution_demo

            run_distribution_demo(outdir / "distributions")

    if args.delete_npz:
        remove_intermediate_npz(outdir)


if __name__ == "__main__":
    main()
