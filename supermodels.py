from __future__ import annotations

from pathlib import Path
import csv

import numpy as np

from triads import TRIADS, MODEL_NAMES
from sampler import sample_params, DISTRIBUTIONS, CODE_TO_DIST
from models import MODELS


SUPERMODELS = {
    # order: (model1, model2, model3a, model3b, model4)
    "SM1": (1.0, 1.0, 1.0, 1.0, 1.0),         # baseline mixture
    "SM2": (1.0, 1.0, 1.0, 1.0, 10.0),        # less Rare-Earth influence
    "SM3": (1.0, 1.0, 1.0, 1.0, 0.1),         # more Rare-Earth influence
    "SM4": (1.0, 1.0, 10.0, 10.0, 1.0),       # less colonization influence
    "SM5": (1.0, 1.0, 0.1, 0.1, 1.0),         # more colonization influence
}

SUPERMODEL_NAMES = list(SUPERMODELS.keys())
SUPERMODEL_COMPONENTS = ["model1", "model2", "model3a", "model3b", "model4"]
GLOBAL_PARAM_NAMES = sorted({name for mid in SUPERMODEL_COMPONENTS for name in TRIADS[mid]})
OPTIONAL_OUTPUT_FIELDS = ("N_active_raw", "N_site_raw", "N_source_raw", "N_gal_cap", "cap_fraction")


def mixture_coefficients(penalties):
    penalties = np.asarray(penalties, dtype=float)
    if penalties.shape != (len(SUPERMODEL_COMPONENTS),):
        raise ValueError(
            f"Expected {len(SUPERMODEL_COMPONENTS)} penalties in supermodel component order {SUPERMODEL_COMPONENTS}, got {penalties}."
        )
    if np.any(penalties <= 0.0):
        raise ValueError(f"All penalty weights must be strictly positive, got {penalties}.")

    inv = 1.0 / penalties
    return inv / np.sum(inv)


def _valid_mask(out):
    L = np.asarray(out["L_active"], dtype=float)
    N = np.asarray(out["N_active"], dtype=float)
    w = np.asarray(out["weight"], dtype=float)
    return np.isfinite(L) & (L > 0.0) & np.isfinite(N) & (N > 0.0) & np.isfinite(w) & (w >= 0.0)


def generate_component_bank(
    samples_per_component: int,
    super_dist: str,
    rng: np.random.Generator,
) -> dict[str, dict[str, np.ndarray | dict[str, np.ndarray]]]:

    if super_dist not in DISTRIBUTIONS:
        raise ValueError(f"Unknown super_dist={super_dist!r}. Allowed: {DISTRIBUTIONS}.")

    samples_per_component = int(samples_per_component)
    if samples_per_component <= 0:
        raise ValueError("samples_per_component must be positive.")

    bank: dict[str, dict] = {}

    for model_id in SUPERMODEL_COMPONENTS:
        params, dist_codes = sample_params(
            TRIADS[model_id],
            samples_per_component,
            super_dist,
            rng,
            return_dist_codes=True,
        )

        model_out = MODELS[model_id](params)
        ok = _valid_mask(model_out)
        if not np.any(ok):
            raise ValueError(f"No valid component samples generated for {model_id}.")

        filtered_out = {}
        for key, value in model_out.items():
            arr = np.asarray(value)
            if arr.shape == ():
                filtered_out[key] = arr
            elif arr.shape[0] == ok.shape[0]:
                filtered_out[key] = arr[ok]
            else:
                filtered_out[key] = arr

        filtered_params = {}
        for key, value in params.items():
            arr = np.asarray(value)
            filtered_params[key] = arr[ok] if arr.shape[0] == ok.shape[0] else arr

        filtered_dist_codes = {}
        for key, value in dist_codes.items():
            arr = np.asarray(value)
            filtered_dist_codes[key] = arr[ok] if arr.shape[0] == ok.shape[0] else arr

        bank[model_id] = {
            "out": filtered_out,
            "params": filtered_params,
            "dist_codes": filtered_dist_codes,
            "valid_count": int(np.sum(ok)),
        }

    return bank


def _component_weight(raw_weight: np.ndarray, pi_i: float) -> np.ndarray:
    raw_weight = np.asarray(raw_weight, dtype=float)
    w = raw_weight.copy()
    wsum = float(np.sum(w))
    if wsum <= 0.0 or not np.isfinite(wsum):
        w = np.ones_like(w, dtype=float)
        wsum = float(w.size)
    return float(pi_i) * (w / wsum)


def assemble_supermodel_from_bank(
    super_id: str,
    penalties,
    component_bank: dict[str, dict],
    super_dist: str,
) -> dict[str, np.ndarray]:
    pi = mixture_coefficients(penalties)

    all_L = []
    all_L_surv = []
    all_N_active = []
    all_N_site = []
    all_N_source = []
    all_N_civ = []
    all_weight = []
    all_component = []
    optional_lists = {name: [] for name in OPTIONAL_OUTPUT_FIELDS}
    dist_code_lists = {name: [] for name in GLOBAL_PARAM_NAMES}

    for model_id, pi_i in zip(SUPERMODEL_COMPONENTS, pi):
        if model_id not in component_bank:
            raise KeyError(f"Missing component {model_id!r} in component_bank.")

        out_i = component_bank[model_id]["out"]
        dist_codes_i = component_bank[model_id]["dist_codes"]

        L = np.asarray(out_i["L_active"], dtype=float)
        L_surv = np.asarray(out_i["L_surv"], dtype=float)
        N_active = np.asarray(out_i["N_active"], dtype=float)
        N_site = np.asarray(out_i["N_site"], dtype=float)
        N_source = np.asarray(out_i["N_source"], dtype=float)
        N_civ = np.asarray(out_i["N_civ"], dtype=float)
        w = _component_weight(np.asarray(out_i["weight"], dtype=float), pi_i)

        all_L.append(L)
        all_L_surv.append(L_surv)
        all_N_active.append(N_active)
        all_N_site.append(N_site)
        all_N_source.append(N_source)
        all_N_civ.append(N_civ)
        all_weight.append(w)
        all_component.append(np.full(L.shape, model_id, dtype="U16"))

        for opt_name in OPTIONAL_OUTPUT_FIELDS:
            if opt_name in out_i:
                opt_arr = np.asarray(out_i[opt_name], dtype=float)
                if opt_arr.shape == L.shape:
                    optional_lists[opt_name].append(opt_arr)
                else:
                    optional_lists[opt_name].append(np.full(L.shape, np.nan, dtype=float))
            else:
                optional_lists[opt_name].append(np.full(L.shape, np.nan, dtype=float))

        for param_name in GLOBAL_PARAM_NAMES:
            key = f"dist_{param_name}"
            if key in dist_codes_i:
                codes = np.asarray(dist_codes_i[key], dtype=np.int8)
            else:
                codes = np.full(L.shape, -1, dtype=np.int8)
            dist_code_lists[param_name].append(codes)

    out = {
        "model_id": np.array(super_id),
        "L_active": np.concatenate(all_L),
        "L_surv": np.concatenate(all_L_surv),
        "N_active": np.concatenate(all_N_active),
        "N_site": np.concatenate(all_N_site),
        "N_source": np.concatenate(all_N_source),
        "N_civ": np.concatenate(all_N_civ),
        "weight": np.concatenate(all_weight),
        "component_model": np.concatenate(all_component),
        "penalties": np.asarray(penalties, dtype=float),
        "pi": pi,
        "component_order": np.asarray(SUPERMODEL_COMPONENTS, dtype="U16"),
        "super_dist": np.array(super_dist),
        "dist_code_meaning": CODE_TO_DIST,
        "common_component_bank": np.array(True),
    }

    for opt_name, lists in optional_lists.items():
        if lists:
            out[opt_name] = np.concatenate(lists)

    for param_name, lists in dist_code_lists.items():
        if lists:
            out[f"dist_{param_name}"] = np.concatenate(lists).astype(np.int8, copy=False)

    return out


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


def weighted_prob(condition, w) -> float:
    condition = np.asarray(condition, dtype=bool)
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(w) & (w >= 0.0)
    if not np.any(ok):
        return float("nan")
    if float(np.sum(w[ok])) <= 0.0:
        return float(np.mean(condition[ok]))
    return float(np.sum(w[ok] * condition[ok]) / np.sum(w[ok]))


def summarize_supermodel(super_id: str, out: dict[str, np.ndarray]) -> dict[str, float | str]:
    L = np.asarray(out["L_active"], dtype=float)
    L_surv = np.asarray(out["L_surv"], dtype=float)
    N_active = np.asarray(out["N_active"], dtype=float)
    N_source = np.asarray(out["N_source"], dtype=float)
    N_site = np.asarray(out["N_site"], dtype=float)
    w = np.asarray(out["weight"], dtype=float)
    pi = np.asarray(out["pi"], dtype=float)

    logL = np.log10(np.maximum(L, 1.0e-300))
    logL_surv = np.log10(np.maximum(L_surv, 1.0e-300))
    logN_active = np.log10(np.maximum(N_active, 1.0e-300))
    logN_source = np.log10(np.maximum(N_source, 1.0e-300))
    logN_site = np.log10(np.maximum(N_site, 1.0e-300))

    q = np.array([0.10, 0.50, 0.90])
    logL_q = weighted_quantile(logL, w, q)
    logL_surv_q = weighted_quantile(logL_surv, w, q)
    logN_active_q = weighted_quantile(logN_active, w, q)
    logN_source_q = weighted_quantile(logN_source, w, q)
    logN_site_q = weighted_quantile(logN_site, w, q)

    row: dict[str, float | str] = {
        "supermodel": super_id,
        "n_samples": int(L.size),
        "sum_weight": float(np.sum(w)),
        "logL_q10": float(logL_q[0]),
        "logL_q50": float(logL_q[1]),
        "logL_q90": float(logL_q[2]),
        "L_q10": float(10.0 ** logL_q[0]),
        "L_q50": float(10.0 ** logL_q[1]),
        "L_q90": float(10.0 ** logL_q[2]),
        "logL_surv_q10": float(logL_surv_q[0]),
        "logL_surv_q50": float(logL_surv_q[1]),
        "logL_surv_q90": float(logL_surv_q[2]),
        "P_L_gt_1e3": weighted_prob(L > 1.0e3, w),
        "P_L_gt_1e4": weighted_prob(L > 1.0e4, w),
        "P_L_gt_1e5": weighted_prob(L > 1.0e5, w),
        "P_L_gt_1e6": weighted_prob(L > 1.0e6, w),
        "P_N_active_gt_1e3": weighted_prob(N_active > 1.0e3, w),
        "P_N_source_gt_1e3": weighted_prob(N_source > 1.0e3, w),
        "P_N_site_gt_1e3": weighted_prob(N_site > 1.0e3, w),
        "logN_active_q10": float(logN_active_q[0]),
        "logN_active_q50": float(logN_active_q[1]),
        "logN_active_q90": float(logN_active_q[2]),
        "logN_source_q10": float(logN_source_q[0]),
        "logN_source_q50": float(logN_source_q[1]),
        "logN_source_q90": float(logN_source_q[2]),
        "logN_site_q10": float(logN_site_q[0]),
        "logN_site_q50": float(logN_site_q[1]),
        "logN_site_q90": float(logN_site_q[2]),
    }

    for model_id, pi_i in zip(SUPERMODEL_COMPONENTS, pi):
        row[f"pi_{model_id}"] = float(pi_i)

    return row


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_supermodel(
    super_id: str,
    penalties,
    samples_per_component: int,
    super_dist: str,
    rng: np.random.Generator,
    out_path: Path,
):

    bank = generate_component_bank(samples_per_component, super_dist, rng)
    out = assemble_supermodel_from_bank(super_id, penalties, bank, super_dist)
    np.savez_compressed(out_path, **out)
    return out_path


def generate_all_supermodels(
    outdir: Path,
    samples_per_component: int,
    super_dist: str,
    rng: np.random.Generator,
):

    outdir.mkdir(parents=True, exist_ok=True)

    component_bank = generate_component_bank(
        samples_per_component=samples_per_component,
        super_dist=super_dist,
        rng=rng,
    )

    paths = []
    summary_rows = []

    for super_id in SUPERMODEL_NAMES:
        path = outdir / f"{super_id}.npz"
        out = assemble_supermodel_from_bank(
            super_id=super_id,
            penalties=SUPERMODELS[super_id],
            component_bank=component_bank,
            super_dist=super_dist,
        )
        np.savez_compressed(path, **out)
        paths.append(path)
        summary_rows.append(summarize_supermodel(super_id, out))

    write_summary_csv(outdir / "supermodel_summary.csv", summary_rows)
    return paths
