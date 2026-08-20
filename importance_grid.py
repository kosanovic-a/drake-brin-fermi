from __future__ import annotations


import argparse
import time
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")

from figure_style import configure_matplotlib_for_latex, savefig_publication
configure_matplotlib_for_latex()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from triads import TRIADS, MODEL_NAMES
from sampler import sample_params, DISTRIBUTIONS
from models import MODELS
from supermodels import SUPERMODELS, SUPERMODEL_NAMES, mixture_coefficients


BASE_MODEL_KEYS = list(MODEL_NAMES)
SUPERMODEL_KEYS = list(SUPERMODEL_NAMES)

MODEL_SHORT_LABELS = {
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

MODEL_INDEX = {
    "model1": 1.0,
    "model2": 2.0,
    "model3a": 3.0,
    "model3b": 3.5,
    "model4": 4.0,
}


FEATURE_LABELS = {
    "WhichModel": "Model",
    "logN": r"$N$",

    "R_star": r"$R_\ast$",
    "fp": r"$f_p$",
    "ne": r"$n_e$",
    "fl": r"$f_l$",
    "fi": r"$f_i$",
    "fa": r"$f_a$",

    "F_astro": r"$F_{\rm astro}$",
    "F_bioactive": r"$F_{\rm bioactive}$",

    "B_density": r"$B$",
    "fg": r"$f_g$",
    "ne_br": r"$n_{e,{\rm br}}$",
    "v": r"$v$",
    "R": r"$R$",

    "M_site": r"$M_{\rm site}$",
    "lambda_det": r"$\lambda_{\rm det}$",
    "p_set": r"$p_{\rm set}$",
    "v_eff": r"$v_{\rm eff}$",
    "t_delay": r"$t_{\rm delay}$",
    "lambda_loc": r"$\lambda_{\rm loc}$",
    "lambda_corr": r"$\lambda_{\rm corr}$",
    "N50": r"$N_{50}$",
    "gamma": r"$\gamma$",
    "T_max": r"$T_{\max}$",
    "N_gal_cap": r"$N_{\rm gal,cap}$",
    "T_det": r"$T_{\rm det}$",
    "T_vis": r"$T_{\rm vis}$",
    "chi_vis": r"$\chi_{\rm vis}$",
    "T50_silence": r"$T_{50,\rm sil}$",
    "eta_silence": r"$\eta_{\rm sil}$",
    "lambda_vis": r"$\lambda_{\rm vis}$",

    "N_star_ne": r"$N_\ast n_e$",
    "fpm": r"$f_{pm}$",
    "fm": r"$f_m$",
    "fj": r"$f_j$",
    "fme": r"$f_{me}$",
    "ne_den": r"$n_{e,{\rm den}}$",
}


EXCLUDE_FEATURES = {
    "alpha",
    "T_obs",
    "Tobs",
    "kappa",
    "kappa_surv",
}

MODEL_FEATURES = {
    "model1": [
        "logN", "R_star", "fp", "ne", "fl", "fi", "fa",
    ],
    "model2": [
        "logN", "F_astro", "F_bioactive",
    ],
    "model3a": [
        "R_star", "fp", "ne", "fl", "fi", "fa",
        "B_density", "fg", "ne_br", "p_set", "v_eff", "R", "t_delay",
        "lambda_loc", "lambda_corr", "N50", "gamma", "T_max", "N_gal_cap",
    ],
    "model3b": [
        "R_star", "fp", "ne", "fl", "fi", "fa",
        "M_site", "N_gal_cap", "lambda_loc", "lambda_corr", "N50", "gamma",
        "T_max", "T_det", "T_vis", "chi_vis", "T50_silence", "eta_silence",
        "lambda_det", "lambda_vis",
    ],
    "model4": [
        "N_star_ne", "fg", "fpm", "fm", "fj", "fme", "R_star", "ne_den",
    ],
}


SUPERMODEL_FEATURES = [
    "WhichModel",

    "R_star", "fp", "ne", "fl", "fi", "fa",
    "F_astro", "F_bioactive",

    "B_density", "fg", "ne_br", "p_set", "v_eff", "R", "t_delay",
    "lambda_loc", "lambda_corr", "N50", "gamma", "T_max", "N_gal_cap",
    "M_site", "T_det", "T_vis", "chi_vis", "T50_silence", "eta_silence", "lambda_det", "lambda_vis",

    "N_star_ne", "fpm", "fm", "fj", "fme", "ne_den",
]

ROW_SPECS = [
    ("Parameter importance", "regression"),
    (r"$P(L<10^3)$", "lt_1e3"),
    (r"$P(10^3<L<10^4)$", "between_1e3_1e4"),
    (r"$P(10^4<L<10^5)$", "between_1e4_1e5"),
    (r"$P(L>10^5)$", "gt_1e5"),
]



def _feature_label(name: str) -> str:
    return FEATURE_LABELS.get(name, name)


def _filter_features(features: list[str]) -> list[str]:
    return [f for f in features if f not in EXCLUDE_FEATURES]


def _safe_log10(x: np.ndarray, floor: float = 1.0e-300) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.log10(np.maximum(x, floor))


def _as_1d(x, n: int | None = None) -> np.ndarray:
    arr = np.asarray(x)
    if arr.shape == ():
        if n is None:
            return arr.reshape(1)
        return np.full(n, arr.item())
    return arr.reshape(-1)


def _target_from_logL(logL: np.ndarray, mode: str) -> np.ndarray:
    if mode == "regression":
        return logL.astype(float)
    if mode == "lt_1e3":
        return (logL < 3.0).astype(int)
    if mode == "between_1e3_1e4":
        return ((logL > 3.0) & (logL < 4.0)).astype(int)
    if mode == "between_1e4_1e5":
        return ((logL > 4.0) & (logL < 5.0)).astype(int)
    if mode == "gt_1e5":
        return (logL > 5.0).astype(int)
    raise ValueError(f"Unknown target mode: {mode}")


def _normalize_importance(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values), values, 0.0)
    values = np.maximum(values, 0.0)
    s = float(np.sum(values))
    if s > 0.0:
        return values / s
    return values



def _sample_component(model_id: str, n: int, dist: str, rng: np.random.Generator) -> tuple[dict, dict]:
    """Sample parameters and evaluate one basic model."""
    params = sample_params(TRIADS[model_id], n, dist, rng, return_dist_codes=False)
    out = MODELS[model_id](params)
    return params, out


def _component_frame(
    model_id: str,
    params: dict,
    out: dict,
    lifetime_key: str = "L_active",
    include_super_columns: bool = False,
    component_weight_scale: float = 1.0,
    normalize_component_weight: bool = False,
) -> pd.DataFrame:
    if lifetime_key not in out:
        raise KeyError(f"Model {model_id} output does not contain {lifetime_key!r}.")

    L = np.asarray(out[lifetime_key], dtype=float)
    n = L.size

    n_key = "N_site" if (str(model_id).startswith("SM") or model_id in {"model3a", "model3b"}) else "N_active"
    N = np.asarray(out.get(n_key, out.get("N_active", params.get("N_active", np.ones(n)))), dtype=float)
    weight = np.asarray(out.get("weight", np.ones(n)), dtype=float)

    if weight.shape == ():
        weight = np.full(n, float(weight))
    else:
        weight = weight.reshape(-1)

    ok = np.isfinite(L) & (L > 0.0) & np.isfinite(N) & (N > 0.0)
    ok &= np.isfinite(weight) & (weight >= 0.0)

    if not np.any(ok):
        raise ValueError(f"No valid positive samples for {model_id}.")

    L = L[ok]
    N = N[ok]
    weight = weight[ok]

    if normalize_component_weight:
        wsum = float(np.sum(weight))
        if wsum <= 0.0 or not np.isfinite(wsum):
            weight = np.ones_like(weight, dtype=float)
            wsum = float(weight.size)
        weight = component_weight_scale * weight / wsum
    else:
        weight = component_weight_scale * weight

    m = L.size
    data: dict[str, np.ndarray] = {
        "logL": _safe_log10(L),
        "logN": _safe_log10(N),
        "weight": weight,
        "WhichModel": np.full(m, MODEL_INDEX[model_id], dtype=float),
    }


    for name, values in params.items():
        if name in EXCLUDE_FEATURES:
            continue
        arr = np.asarray(values, dtype=float).reshape(-1)[ok]
        data[name] = _safe_log10(arr)


    if model_id == "model2":
        pass

    df = pd.DataFrame(data)

    if include_super_columns:
        required_features = _filter_features(SUPERMODEL_FEATURES)
        for col in required_features:
            if col not in df.columns:
                df[col] = 0.0
        cols = ["logL", "weight"] + required_features
    else:
        required_features = _filter_features(MODEL_FEATURES[model_id])
        missing = [col for col in required_features if col not in df.columns]
        if missing:
            raise KeyError(f"Missing feature(s) for {model_id}: {missing}")
        cols = ["logL", "weight"] + required_features

    df = df[cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(f"No finite samples after dataframe cleanup for {model_id}.")

    if float(df["weight"].sum()) <= 0.0:
        df["weight"] = 1.0

    return df


def build_basic_df(model_id: str, n: int, dist: str, seed: int, lifetime_key: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    params, out = _sample_component(model_id, n=n, dist=dist, rng=rng)
    return _component_frame(
        model_id=model_id,
        params=params,
        out=out,
        lifetime_key=lifetime_key,
        include_super_columns=False,
    )


def build_supermodel_df(super_id: str, n_total: int, super_dist: str, seed: int, lifetime_key: str) -> pd.DataFrame:
    if super_id not in SUPERMODELS:
        raise KeyError(f"Unknown supermodel {super_id!r}.")

    penalties = SUPERMODELS[super_id]
    pi = mixture_coefficients(penalties)

    n_components = len(BASE_MODEL_KEYS)
    n_per_component = max(100, int(np.ceil(n_total / n_components)))
    rng = np.random.default_rng(seed)

    frames = []
    for model_id, pi_i in zip(BASE_MODEL_KEYS, pi):
        params, out = _sample_component(
            model_id=model_id,
            n=n_per_component,
            dist=super_dist,
            rng=rng,
        )
        df_i = _component_frame(
            model_id=model_id,
            params=params,
            out=out,
            lifetime_key=lifetime_key,
            include_super_columns=True,
            component_weight_scale=float(pi_i),
            normalize_component_weight=True,
        )
        frames.append(df_i)

    if not frames:
        raise RuntimeError(f"No component frames generated for {super_id}.")

    return pd.concat(frames, ignore_index=True)




def _subsample_for_permutation(
    X: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
    max_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if len(X) <= max_rows:
        return X, y, w
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_rows, replace=False)
    return X.iloc[idx], y[idx], w[idx]


def fit_importance(
    df: pd.DataFrame,
    features: list[str],
    mode: str,
    *,
    test_size: float,
    random_state: int,
    rf_trees: int,
    rf_min_leaf: int,
    rf_n_jobs: int,
    use_permutation: bool,
    perm_repeats: int,
    max_perm_rows: int,
) -> tuple[np.ndarray, float]:
    features = _filter_features(features)
    if len(features) == 0:
        return np.zeros(0, dtype=float), float("nan")

    X = df[features].astype(float)
    y = _target_from_logL(df["logL"].to_numpy(dtype=float), mode)
    w = df["weight"].to_numpy(dtype=float) if "weight" in df.columns else np.ones(len(df))

    ok = np.isfinite(X.to_numpy(dtype=float)).all(axis=1) & np.isfinite(y) & np.isfinite(w) & (w >= 0.0)
    X = X.loc[ok].reset_index(drop=True)
    y = y[ok]
    w = w[ok]

    if len(X) < 50:
        return np.zeros(len(features), dtype=float), float("nan")

    if float(np.sum(w)) <= 0.0:
        w = np.ones_like(w, dtype=float)

    if mode != "regression" and len(np.unique(y)) < 2:
        return np.zeros(len(features), dtype=float), float("nan")

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X,
        y,
        w,
        test_size=test_size,
        random_state=random_state,
    )

    if mode == "regression":
        model = RandomForestRegressor(
            n_estimators=rf_trees,
            min_samples_leaf=rf_min_leaf,
            random_state=random_state,
            n_jobs=rf_n_jobs,
        )
        model.fit(X_train, y_train, sample_weight=w_train)
        pred = model.predict(X_test)
        score = float(r2_score(y_test, pred, sample_weight=w_test))
        scoring = "r2"
    else:
        if len(np.unique(y_train)) < 2:
            return np.zeros(len(features), dtype=float), float("nan")
        model = RandomForestClassifier(
            n_estimators=rf_trees,
            min_samples_leaf=rf_min_leaf,
            random_state=random_state,
            n_jobs=rf_n_jobs,
            class_weight="balanced_subsample",
        )
        model.fit(X_train, y_train, sample_weight=w_train)
        pred = model.predict(X_test)
        try:
            score = float(balanced_accuracy_score(y_test, pred, sample_weight=w_test))
        except Exception:
            score = float(accuracy_score(y_test, pred, sample_weight=w_test))
        scoring = "balanced_accuracy"

    if not use_permutation:
        return _normalize_importance(model.feature_importances_), score

    X_eval, y_eval, w_eval = _subsample_for_permutation(
        X_test,
        y_test,
        w_test,
        max_rows=max_perm_rows,
        seed=random_state + 991,
    )

    try:
        perm = permutation_importance(
            model,
            X_eval,
            y_eval,
            n_repeats=perm_repeats,
            random_state=random_state,
            n_jobs=1,
            scoring=scoring,
            sample_weight=w_eval,
        )
    except TypeError:
        perm = permutation_importance(
            model,
            X_eval,
            y_eval,
            n_repeats=perm_repeats,
            random_state=random_state,
            n_jobs=1,
            scoring=scoring,
        )

    return _normalize_importance(perm.importances_mean), score



def plot_grid(
    model_keys: list[str],
    data_builder: Callable[[str, int], pd.DataFrame],
    feature_getter: Callable[[str], list[str]],
    *,
    outdir: Path,
    out_pdf: str,
    out_csv: str,
    figure_title: str,
    test_size: float,
    random_state: int,
    rf_trees: int,
    rf_min_leaf: int,
    rf_n_jobs: int,
    use_permutation: bool,
    perm_repeats: int,
    max_perm_rows: int,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    n_rows = len(ROW_SPECS)
    n_cols = len(model_keys)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.75 * n_cols, 2.25 * n_rows),
        squeeze=False,
    )

    records: list[dict] = []

    for j, key in enumerate(model_keys):
        print(f"[INFO] Building data for {key} ...")
        df = data_builder(key, j)
        features = _filter_features(feature_getter(key))

        for i, (row_label, mode) in enumerate(ROW_SPECS):
            ax = axes[i, j]
            try:
                imp, score = fit_importance(
                    df,
                    features,
                    mode,
                    test_size=test_size,
                    random_state=random_state,
                    rf_trees=rf_trees,
                    rf_min_leaf=rf_min_leaf,
                    rf_n_jobs=rf_n_jobs,
                    use_permutation=use_permutation,
                    perm_repeats=perm_repeats,
                    max_perm_rows=max_perm_rows,
                )
            except Exception as exc:
                print(f"[WARN] {key}, {mode}: {exc}")
                imp = np.zeros(len(features), dtype=float)
                score = float("nan")

            x = np.arange(len(features))
            ax.bar(x, imp, width=0.72)

            if j == 0:
                ax.set_ylabel(row_label, fontsize=8)

            ax.set_xticks(x)
            ax.set_xticklabels([_feature_label(f) for f in features], rotation=90, fontsize=7)
            ymax = float(np.nanmax(imp)) if imp.size else 0.0
            ax.set_ylim(0.0, max(0.05, 1.15 * ymax))
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(axis="y", alpha=0.18, linewidth=0.5)

            for f, val in zip(features, imp):
                records.append(
                    {
                        "model": key,
                        "target": row_label,
                        "mode": mode,
                        "feature_internal": f,
                        "feature_label": _feature_label(f),
                        "importance": float(val),
                        "score": score,
                        "n_rows": int(len(df)),
                        "weighted": True,
                        "importance_method": "permutation" if use_permutation else "rf_impurity",
                    }
                )

    fig.tight_layout()
    savefig_publication(fig, outdir / out_pdf, dpi=220, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame.from_records(records).to_csv(outdir / out_csv, index=False)




def run_importance_grid(
    *,
    out: str | Path = "outputs",
    dist: str = "lognormal",
    super_dist: str = "mixed",
    n_basic: int = 80_000,
    n_super_total: int = 120_000,
    lifetime_key: str = "L_active",
    seed: int = 12345,
    test_size: float = 0.25,
    rf_trees: int = 300,
    rf_min_leaf: int = 20,
    rf_n_jobs: int = -1,
    use_permutation: bool = True,
    perm_repeats: int = 8,
    max_perm_rows: int = 25_000,
) -> None:
    if dist not in DISTRIBUTIONS:
        raise ValueError(f"Unknown dist={dist!r}. Allowed: {DISTRIBUTIONS}")
    if super_dist not in DISTRIBUTIONS:
        raise ValueError(f"Unknown super_dist={super_dist!r}. Allowed: {DISTRIBUTIONS}")

    t0 = time.time()
    out = Path(out)
    grid_dir = out / "importance_grid"
    grid_dir.mkdir(parents=True, exist_ok=True)

    base_seed = int(seed) + 400_000

    def build_basic(model_id: str, offset: int) -> pd.DataFrame:
        return build_basic_df(
            model_id=model_id,
            n=int(n_basic),
            dist=dist,
            seed=base_seed + 101 * offset,
            lifetime_key=lifetime_key,
        )

    def basic_features(model_id: str) -> list[str]:
        return _filter_features(MODEL_FEATURES[model_id])

    def build_super(super_id: str, offset: int) -> pd.DataFrame:
        return build_supermodel_df(
            super_id=super_id,
            n_total=int(n_super_total),
            super_dist=super_dist,
            seed=base_seed + 10_000,
            lifetime_key=lifetime_key,
        )

    def super_features(_super_id: str) -> list[str]:
        return _filter_features(SUPERMODEL_FEATURES)

    print("[INFO] Drawing importance grid for basic models ...")
    plot_grid(
        BASE_MODEL_KEYS,
        build_basic,
        basic_features,
        outdir=grid_dir,
        out_pdf="importance_grid_basic.pdf",
        out_csv="importance_scores_basic.csv",
        figure_title="Parameter importance: basic models",
        test_size=test_size,
        random_state=seed,
        rf_trees=rf_trees,
        rf_min_leaf=rf_min_leaf,
        rf_n_jobs=rf_n_jobs,
        use_permutation=use_permutation,
        perm_repeats=perm_repeats,
        max_perm_rows=max_perm_rows,
    )

    print("[INFO] Drawing importance grid for supermodels ...")
    plot_grid(
        SUPERMODEL_KEYS,
        build_super,
        super_features,
        outdir=grid_dir,
        out_pdf="importance_grid_supermodels.pdf",
        out_csv="importance_scores_supermodels.csv",
        figure_title="Parameter importance: supermodels",
        test_size=test_size,
        random_state=seed,
        rf_trees=rf_trees,
        rf_min_leaf=rf_min_leaf,
        rf_n_jobs=rf_n_jobs,
        use_permutation=use_permutation,
        perm_repeats=perm_repeats,
        max_perm_rows=max_perm_rows,
    )

    print(f"[OK] Importance grid finished in {time.time() - t0:.1f} s")
    print(f"[OK] Outputs written to: {grid_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parameter-importance grids for CIVILIZACIJE.")
    parser.add_argument("--out", default="outputs", help="Output directory. Default: outputs")
    parser.add_argument("--dist", default="lognormal", choices=DISTRIBUTIONS, help="Distribution for basic models.")
    parser.add_argument("--super-dist", default="mixed", choices=DISTRIBUTIONS, help="Distribution for supermodel components.")
    parser.add_argument("--n-basic", type=int, default=80_000, help="Samples per basic model.")
    parser.add_argument("--n-super-total", type=int, default=120_000, help="Approximate samples per supermodel.")
    parser.add_argument("--lifetime", default="L_active", choices=["L_active", "L_surv"], help="Lifetime target.")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--rf-trees", type=int, default=300)
    parser.add_argument("--rf-min-leaf", type=int, default=20)
    parser.add_argument("--rf-n-jobs", type=int, default=-1, help="Parallel jobs for RandomForest. Permutation importance is kept single-job to avoid nested joblib overhead.")
    parser.add_argument("--perm-repeats", type=int, default=8)
    parser.add_argument("--max-perm-rows", type=int, default=25_000)
    parser.add_argument("--impurity", action="store_true", help="Use RF impurity importance instead of permutation importance.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_importance_grid(
        out=args.out,
        dist=args.dist,
        super_dist=args.super_dist,
        n_basic=args.n_basic,
        n_super_total=args.n_super_total,
        lifetime_key=args.lifetime,
        seed=args.seed,
        test_size=args.test_size,
        rf_trees=args.rf_trees,
        rf_min_leaf=args.rf_min_leaf,
        rf_n_jobs=args.rf_n_jobs,
        use_permutation=not args.impurity,
        perm_repeats=args.perm_repeats,
        max_perm_rows=args.max_perm_rows,
    )


if __name__ == "__main__":
    main()
