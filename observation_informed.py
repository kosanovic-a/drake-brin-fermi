from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from bayesian_reweighting import MarginalCalibrationResult, calibrate_positive_marginal, effective_sample_size, estimate_histogram_density, normalize_log_weights, weighted_mean, weighted_quantile
from figure_style import configure_matplotlib_for_latex, savefig_publication
from models import MODELS
from observational_constraints import BRYSON_ETA_HZ_SCENARIOS, AsymmetricLog10Density, get_eta_hz_constraint
from sampler import DISTRIBUTIONS, sample_params, sample_triad, sample_triad_mixed
from seti_observation import MWA_VELA_2020, semantic_population_from_output, survey_log_likelihood
from supermodels import SUPERMODELS, SUPERMODEL_COMPONENTS, mixture_coefficients
from triad_sensitivity import DEFAULT_VARIANTS, WEAK_PARAMETER_GROUPS, apply_variant, distribution_distance_summary
from triads import MODEL_NAMES, TRIADS
configure_matplotlib_for_latex()
MODEL_LABELS = {'model1': 'Model I', 'model2': 'Model II', 'model3a': 'Model IIIA', 'model3b': 'Model IIIB-M', 'model4': 'Model IV'}

@dataclass
class Ensemble:
    model_id: str
    params: dict[str, np.ndarray]
    outputs: dict[str, np.ndarray]
    dist_codes: dict[str, np.ndarray]

    @property
    def n(self) -> int:
        return int(np.asarray(self.outputs['L_active']).size)

class EmpiricalLog10Density:

    def __init__(self, density) -> None:
        self.density = density

    def logpdf_log10(self, x):
        return self.density.logpdf(x)

def _filter_ensemble(model_id: str, params: dict[str, np.ndarray], outputs: dict[str, np.ndarray], dist_codes: dict[str, np.ndarray]) -> Ensemble:
    L = np.asarray(outputs['L_active'], dtype=float)
    N = np.asarray(outputs['N_active'], dtype=float)
    w = np.asarray(outputs.get('weight', np.ones_like(L)), dtype=float)
    ok = np.isfinite(L) & (L > 0.0) & np.isfinite(N) & (N > 0.0)
    ok &= np.isfinite(w) & (w >= 0.0)
    if not np.any(ok):
        raise ValueError(f'No valid samples generated for {model_id}.')

    def filt(mapping):
        result = {}
        for key, value in mapping.items():
            arr = np.asarray(value)
            if arr.shape == ():
                result[key] = arr
            elif arr.shape[0] == ok.size:
                result[key] = arr[ok]
            else:
                result[key] = arr
        return result
    return Ensemble(model_id=model_id, params=filt(params), outputs=filt(outputs), dist_codes=filt(dist_codes))

def generate_ensemble(model_id: str, *, n: int, dist: str, seed: int, triads_override: dict[str, list[float]] | None=None) -> Ensemble:
    rng = np.random.default_rng(seed)
    triads = TRIADS[model_id] if triads_override is None else triads_override
    params, dist_codes = sample_params(triads, int(n), dist, rng, return_dist_codes=True)
    outputs = MODELS[model_id](params)
    return _filter_ensemble(model_id, params, outputs, dist_codes)

def generate_observation_informed_ensemble(baseline: Ensemble, target: AsymmetricLog10Density, *, dist: str, seed: int) -> tuple[Ensemble, str]:
    model_id = baseline.model_id
    if model_id == 'model4':
        return (baseline, 'not_applied_semantic_mismatch_N_star_ne')
    rng = np.random.default_rng(seed)
    params = {key: np.asarray(value).copy() for key, value in baseline.params.items()}
    dist_codes = {key: np.asarray(value).copy() for key, value in baseline.dist_codes.items()}
    eta = target.sample(baseline.n, rng)
    if model_id in {'model1', 'model3a', 'model3b'}:
        params['fp_baseline_unused'] = params.pop('fp')
        params['ne_baseline_unused'] = params.pop('ne')
        if 'dist_fp' in dist_codes:
            dist_codes['dist_fp_baseline_unused'] = dist_codes.pop('dist_fp')
        if 'dist_ne' in dist_codes:
            dist_codes['dist_ne_baseline_unused'] = dist_codes.pop('dist_ne')
        params['eta_hz'] = eta
        dist_codes['dist_eta_hz'] = np.full(baseline.n, -2, dtype=np.int8)
        outputs = MODELS[model_id](params)
        return (_filter_ensemble(model_id, params, outputs, dist_codes), 'direct_eta_hz_external_posterior')
    if model_id == 'model2':
        rstar = _draw_rstar_target(baseline.n, dist, rng)
        params['eta_hz'] = eta
        params['R_star_obs'] = rstar
        params['F_astro'] = rstar * eta
        dist_codes['dist_eta_hz'] = np.full(baseline.n, -2, dtype=np.int8)
        dist_codes['dist_R_star_obs'] = np.full(baseline.n, -3, dtype=np.int8)
        outputs = MODELS[model_id](params)
        return (_filter_ensemble(model_id, params, outputs, dist_codes), 'direct_F_astro_equals_R_star_times_eta_hz')
    raise KeyError(f'Unsupported model {model_id!r}.')

def _base_weights(ensemble: Ensemble) -> np.ndarray:
    raw = np.asarray(ensemble.outputs.get('weight', np.ones(ensemble.n)), dtype=float)
    if raw.shape != (ensemble.n,) or not np.all(np.isfinite(raw)) or np.any(raw < 0.0):
        raw = np.ones(ensemble.n, dtype=float)
    if float(np.sum(raw)) <= 0.0:
        raw = np.ones(ensemble.n, dtype=float)
    return raw / np.sum(raw)

def _draw_rstar_target(n: int, dist: str, rng: np.random.Generator) -> np.ndarray:
    triad = TRIADS['model1']['R_star']
    if dist == 'mixed':
        values, _ = sample_triad_mixed(triad, n, rng)
        return values
    return sample_triad(triad, n, dist, rng)

def exoplanet_calibration(ensemble: Ensemble, target: AsymmetricLog10Density, *, dist: str, seed: int, density_bins: int=180) -> tuple[np.ndarray, MarginalCalibrationResult | None, str, np.ndarray | None]:
    base = _base_weights(ensemble)
    model_id = ensemble.model_id
    if model_id in {'model1', 'model3a', 'model3b'}:
        eta = np.asarray(ensemble.params['fp'], dtype=float) * np.asarray(ensemble.params['ne'], dtype=float)
        result = calibrate_positive_marginal(eta, target, base_weights=base, bins=density_bins)
        return (result.weights, result, 'eta_hz_equals_fp_times_ne', eta)
    if model_id == 'model2':
        rng = np.random.default_rng(seed)
        n_target = max(200000, ensemble.n)
        rstar = _draw_rstar_target(n_target, dist, rng)
        eta_target = target.sample(n_target, rng)
        f_target = rstar * eta_target
        target_hist = estimate_histogram_density(np.log10(f_target), bins=density_bins)
        result = calibrate_positive_marginal(np.asarray(ensemble.params['F_astro'], dtype=float), EmpiricalLog10Density(target_hist), base_weights=base, bins=density_bins)
        return (result.weights, result, 'F_astro_equals_R_star_times_eta_hz', np.asarray(ensemble.params['F_astro'], dtype=float))
    return (base, None, 'not_applied_semantic_mismatch_N_star_ne', None)

def _nplot(ensemble: Ensemble) -> np.ndarray:
    if ensemble.model_id in {'model3a', 'model3b'}:
        return np.asarray(ensemble.outputs['N_site'], dtype=float)
    return np.asarray(ensemble.outputs['N_active'], dtype=float)

def _summary_records(ensemble: Ensemble, stage: str, weights: np.ndarray, calibration_note: str) -> list[dict[str, object]]:
    metrics: dict[str, np.ndarray] = {'log10_L_active': np.log10(np.asarray(ensemble.outputs['L_active'], dtype=float)), 'log10_N_plot': np.log10(_nplot(ensemble)), 'log10_N_source': np.log10(np.asarray(ensemble.outputs['N_source'], dtype=float)), 'log10_N_site': np.log10(np.asarray(ensemble.outputs['N_site'], dtype=float))}
    if 'eta_hz' in ensemble.params:
        metrics['log10_eta_hz'] = np.log10(np.asarray(ensemble.params['eta_hz'], dtype=float))
    elif 'fp' in ensemble.params and 'ne' in ensemble.params:
        metrics['log10_eta_hz'] = np.log10(np.asarray(ensemble.params['fp'], dtype=float) * np.asarray(ensemble.params['ne'], dtype=float))
    if 'F_astro' in ensemble.params:
        metrics['log10_F_astro'] = np.log10(np.asarray(ensemble.params['F_astro'], dtype=float))
    if 'C_D' in ensemble.outputs:
        metrics['log10_C_D'] = np.log10(np.asarray(ensemble.outputs['C_D'], dtype=float))
    rows = []
    ess = effective_sample_size(weights)
    for metric, values in metrics.items():
        q05, q50, q95 = weighted_quantile(values, weights, [0.05, 0.5, 0.95])
        rows.append({'model': ensemble.model_id, 'model_label': MODEL_LABELS[ensemble.model_id], 'stage': stage, 'metric': metric, 'mean': weighted_mean(values, weights), 'q05': float(q05), 'q50': float(q50), 'q95': float(q95), 'n': ensemble.n, 'ess': ess, 'ess_fraction': ess / ensemble.n, 'calibration_note': calibration_note})
    return rows

def _regime_masses(ensemble: Ensemble, weights: np.ndarray) -> dict[str, float]:
    L = np.asarray(ensemble.outputs['L_active'], dtype=float)
    N = _nplot(ensemble)
    r_exp = (L >= 10000.0) & (N >= 100.0)
    masks = {'short': L < 1000.0, 'medium': (L >= 1000.0) & (L < 100000.0) & ~r_exp, 'long_localized': (L >= 100000.0) & (N < 100.0), 'multi_site': r_exp}
    total = float(np.sum(weights))
    return {name: float(np.sum(weights[mask]) / total) for name, mask in masks.items()}

def _save_stage_npz(path: Path, ensemble: Ensemble, stage_weights: dict[str, np.ndarray]) -> None:
    payload: dict[str, np.ndarray] = {}
    payload.update(ensemble.outputs)
    for key, value in ensemble.params.items():
        out_key = key if key not in payload else f'param_{key}'
        payload[out_key] = value
    payload.update(ensemble.dist_codes)
    payload['model_id'] = np.array(ensemble.model_id)
    payload['weight_baseline'] = _base_weights(ensemble)
    for stage, weights in stage_weights.items():
        payload[f'weight_{stage}'] = np.asarray(weights, dtype=float)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)

def _density_curve(values, weights, edges):
    counts, _ = np.histogram(values, bins=edges, weights=weights)
    counts = counts.astype(float)
    dx = edges[1] - edges[0]
    if np.sum(counts) > 0:
        counts /= np.sum(counts) * dx
    return (0.5 * (edges[:-1] + edges[1:]), counts)

def _eta_values(ensemble: Ensemble) -> np.ndarray:
    if 'eta_hz' in ensemble.params:
        return np.asarray(ensemble.params['eta_hz'], dtype=float)
    if 'fp' in ensemble.params and 'ne' in ensemble.params:
        return np.asarray(ensemble.params['fp'], dtype=float) * np.asarray(ensemble.params['ne'], dtype=float)
    raise KeyError(f'{ensemble.model_id} has no eta_HZ representation.')

def plot_marginal_calibration(baseline: Ensemble, observation_informed: Ensemble, external_target: AsymmetricLog10Density, path: Path) -> None:
    if baseline.model_id not in {'model1', 'model3a', 'model3b'}:
        return
    x0 = np.log10(_eta_values(baseline))
    x1 = np.log10(_eta_values(observation_informed))
    lo = min(float(np.min(x0)), float(np.min(x1)), np.log10(external_target.lower_68) - 0.4)
    hi = max(float(np.max(x0)), float(np.max(x1)), np.log10(external_target.upper_68) + 0.4)
    edges = np.linspace(lo, hi, 150)
    centers, prior = _density_curve(x0, _base_weights(baseline), edges)
    _, posterior = _density_curve(x1, _base_weights(observation_informed), edges)
    target = external_target.pdf_log10(centers)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(centers, prior, label='Baseline')
    ax.plot(centers, posterior, label='Observation-informed')
    ax.plot(centers, target, linestyle='--', label='External marginal')
    ax.set_xlabel('$\\log_{10}(\\eta_{\\mathrm{HZ}})$')
    ax.set_ylabel('Density')
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    savefig_publication(fig, path)
    plt.close(fig)

def plot_prior_posterior_heatmaps(baseline: Ensemble, observation_informed: Ensemble, path: Path, bins: int=80) -> None:
    datasets = []
    for ensemble in (baseline, observation_informed):
        datasets.append((np.log10(np.asarray(ensemble.outputs['L_active'], dtype=float)), np.log10(_nplot(ensemble)), _base_weights(ensemble)))
    all_logL = np.concatenate([item[0] for item in datasets])
    all_logN = np.concatenate([item[1] for item in datasets])
    xlo, xhi = np.quantile(all_logL, [0.002, 0.998])
    ylo, yhi = np.quantile(all_logN, [0.002, 0.998])
    ylo = max(0.0, float(ylo))
    xr = (float(xlo), float(xhi))
    yr = (ylo, float(yhi))
    matrices = []
    edges = None
    for logL, logN, weights in datasets:
        H, xe, ye = np.histogram2d(logL, logN, bins=bins, range=[xr, yr], weights=weights)
        H = gaussian_filter(H, sigma=1.5, mode='nearest')
        dx = xe[1] - xe[0]
        dy = ye[1] - ye[0]
        if np.sum(H) > 0:
            H /= np.sum(H) * dx * dy
        matrices.append(H)
        edges = (xe, ye)
    vmax = max((float(np.max(h)) for h in matrices))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharex=True, sharey=True)
    im = None
    for ax, H in zip(axes, matrices):
        xe, ye = edges
        im = ax.imshow(H.T, origin='lower', extent=[xe[0], xe[-1], ye[0], ye[-1]], aspect='auto', vmin=0.0, vmax=vmax)
        ax.set_xlabel('$\\log_{10}(L_{\\mathrm{active}}/\\mathrm{yr})$')
    axes[0].set_ylabel('$\\log_{10}(N_{\\mathrm{plot}})$')
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        cbar.set_label('Density')
    fig.subplots_adjust(left=0.08, right=0.91, bottom=0.16, top=0.98, wspace=0.08)
    savefig_publication(fig, path)
    plt.close(fig)

def _parameter_support_rows() -> list[dict[str, str]]:
    rows = []
    for model_id in MODEL_NAMES:
        for parameter in TRIADS[model_id]:
            if parameter in {'fp', 'ne'}:
                status = 'partially_observation_constrained'
                rationale = 'Kepler occurrence information is applied to the product eta_HZ=fp*ne.'
            elif parameter == 'R_star':
                status = 'literature_anchored_not_directly_calibrated'
                rationale = "Published mass SFR is not identical to the model's relevant-system formation rate."
            elif parameter in WEAK_PARAMETER_GROUPS.get(model_id, ()):
                status = 'scenario_or_weakly_constrained'
                rationale = 'No direct population-level calibration; included in prior-sensitivity analysis.'
            elif parameter == 'N_star_ne':
                status = 'aggregate_partially_constrained'
                rationale = 'Different aggregate stellar-population semantics prevent direct eta_HZ substitution.'
            else:
                status = 'derived_or_literature_anchored'
                rationale = 'Retained under the baseline parameter specification.'
            rows.append({'model': model_id, 'parameter': parameter, 'status': status, 'rationale': rationale})
    return rows

def _run_seti_sensitivity(ensemble: Ensemble, base_weights: np.ndarray, *, logq_grid: np.ndarray, n_galactic_targets: float, settlement_bridge: str, allow_model3b: bool) -> list[dict[str, object]]:
    if ensemble.model_id == 'model3b' and (not allow_model3b):
        return [{'model': ensemble.model_id, 'log10_q_eff': np.nan, 'ess': effective_sample_size(base_weights), 'ess_fraction': effective_sample_size(base_weights) / ensemble.n, 'posterior_logL_median': weighted_quantile(np.log10(ensemble.outputs['L_active']), base_weights, [0.5])[0], 'status': 'skipped_to_avoid_double_use_of_silence_conditioning'}]
    population = semantic_population_from_output(ensemble.outputs, ensemble.model_id, settlement_bridge=settlement_bridge)
    logL = np.log10(np.asarray(ensemble.outputs['L_active'], dtype=float))
    records = []
    for logq in logq_grid:
        log_like = survey_log_likelihood(population, MWA_VELA_2020, q_eff=float(10.0 ** logq), n_galactic_targets=n_galactic_targets)
        weights, diag = normalize_log_weights(log_like, base_weights)
        med = float(weighted_quantile(logL, weights, [0.5])[0])
        records.append({'model': ensemble.model_id, 'log10_q_eff': float(logq), 'ess': diag.ess, 'ess_fraction': diag.ess_fraction, 'posterior_logL_median': med, 'max_weight': diag.max_weight, 'status': 'conditional_sensitivity_not_calibrated_posterior'})
    return records

def plot_seti_sensitivity(df: pd.DataFrame, path: Path) -> None:
    valid = df[np.isfinite(df['log10_q_eff'])].copy()
    if valid.empty:
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for model_id, group in valid.groupby('model'):
        ax.plot(group['log10_q_eff'], group['ess_fraction'], marker='o', markersize=2.5, label=MODEL_LABELS[model_id])
    ax.set_xlabel('$\\log_{10}(q_{\\mathrm{eff}})$')
    ax.set_ylabel('$N_{\\mathrm{eff}}/N$')
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    savefig_publication(fig, path)
    plt.close(fig)

def _run_prior_sensitivity(*, model_id: str, n: int, seed: int, baseline_dist: str) -> list[dict[str, object]]:
    baseline = generate_ensemble(model_id, n=n, dist=baseline_dist, seed=seed)
    base_logL = np.log10(np.asarray(baseline.outputs['L_active'], dtype=float))
    base_logN = np.log10(_nplot(baseline))
    records: list[dict[str, object]] = []

    def add_record(kind: str, variant: str, ensemble: Ensemble):
        logL = np.log10(np.asarray(ensemble.outputs['L_active'], dtype=float))
        logN = np.log10(_nplot(ensemble))
        dL = distribution_distance_summary(base_logL, logL)
        dN = distribution_distance_summary(base_logN, logN)
        qL = np.quantile(logL, [0.05, 0.5, 0.95])
        qN = np.quantile(logN, [0.05, 0.5, 0.95])
        records.append({'model': model_id, 'kind': kind, 'variant': variant, 'logL_q05': qL[0], 'logL_q50': qL[1], 'logL_q95': qL[2], 'logN_q05': qN[0], 'logN_q50': qN[1], 'logN_q95': qN[2], 'logL_wasserstein': dL['wasserstein'], 'logL_jensen_shannon': dL['jensen_shannon'], 'logN_wasserstein': dN['wasserstein'], 'logN_jensen_shannon': dN['jensen_shannon']})
    add_record('reference', baseline_dist, baseline)
    for i, dist in enumerate(('lognormal', 'loglinear', 'gauss')):
        if dist == baseline_dist:
            continue
        ens = generate_ensemble(model_id, n=n, dist=dist, seed=seed + 101 * (i + 1))
        add_record('distribution_family', dist, ens)
    weak = WEAK_PARAMETER_GROUPS[model_id]
    for i, variant in enumerate(DEFAULT_VARIANTS):
        if variant.name == 'baseline':
            continue
        triads_variant = apply_variant(TRIADS[model_id], weak, variant)
        ens = generate_ensemble(model_id, n=n, dist=baseline_dist, seed=seed + 10000 + 101 * i, triads_override=triads_variant)
        add_record('weak_triad_perturbation', variant.name, ens)
    return records

def _assemble_supermodels(outdir: Path, ensembles: dict[str, Ensemble], stage: str) -> list[dict[str, object]]:
    rows = []
    for super_id, penalties in SUPERMODELS.items():
        pi = mixture_coefficients(penalties)
        payload = {'model_id': np.array(super_id), 'component_model': [], 'L_active': [], 'L_surv': [], 'N_active': [], 'N_site': [], 'N_source': [], 'N_civ': [], 'weight': [], 'pi': pi, 'penalties': np.asarray(penalties, dtype=float)}
        for model_id, pi_i in zip(SUPERMODEL_COMPONENTS, pi):
            ens = ensembles[model_id]
            w = _base_weights(ens) * float(pi_i)
            for key in ('L_active', 'L_surv', 'N_active', 'N_site', 'N_source', 'N_civ'):
                payload[key].append(np.asarray(ens.outputs[key]))
            payload['weight'].append(w)
            payload['component_model'].append(np.full(ens.n, model_id, dtype='U16'))
        for key in ('component_model', 'L_active', 'L_surv', 'N_active', 'N_site', 'N_source', 'N_civ', 'weight'):
            payload[key] = np.concatenate(payload[key])
        path = outdir / 'supermodels' / f'{super_id}_{stage}.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **payload)
        w = payload['weight']
        logL = np.log10(payload['L_active'])
        logN = np.log10(payload['N_active'])
        qL = weighted_quantile(logL, w, [0.05, 0.5, 0.95])
        qN = weighted_quantile(logN, w, [0.05, 0.5, 0.95])
        rows.append({'supermodel': super_id, 'stage': stage, 'logL_q05': qL[0], 'logL_q50': qL[1], 'logL_q95': qL[2], 'logN_q05': qN[0], 'logN_q50': qN[1], 'logN_q95': qN[2]})
    return rows

def run_comment2(*, outdir: Path, samples: int, dist: str, seed: int, eta_scenario: str, sensitivity_samples: int, run_seti_sensitivity: bool, seti_logq_min: float, seti_logq_max: float, seti_grid_points: int, n_galactic_targets: float, settlement_bridge: str, allow_model3b_seti: bool) -> None:
    if dist not in DISTRIBUTIONS:
        raise ValueError(f'Unknown distribution {dist!r}.')
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'figures').mkdir(exist_ok=True)
    target = get_eta_hz_constraint(eta_scenario)
    baseline_ensembles: dict[str, Ensemble] = {}
    observation_ensembles: dict[str, Ensemble] = {}
    summary_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    regime_rows: list[dict[str, object]] = []
    for index, model_id in enumerate(MODEL_NAMES):
        print(f'[INFO] Generating baseline and observation-informed {model_id} ...')
        baseline = generate_ensemble(model_id, n=samples, dist=dist, seed=seed + 1009 * index)
        observation, observation_note = generate_observation_informed_ensemble(baseline, target, dist=dist, seed=seed + 20000 + 1009 * index)
        baseline_ensembles[model_id] = baseline
        observation_ensembles[model_id] = observation
        w0 = _base_weights(baseline)
        w_obs = _base_weights(observation)
        summary_rows.extend(_summary_records(baseline, 'baseline', w0, 'bounded_scenario'))
        summary_rows.extend(_summary_records(observation, 'observation_informed', w_obs, observation_note))
        regime_rows.append({'model': model_id, 'stage': 'baseline', **_regime_masses(baseline, w0)})
        regime_rows.append({'model': model_id, 'stage': 'observation_informed', **_regime_masses(observation, w_obs)})
        ratio_weights, calibration, ratio_note, _ = exoplanet_calibration(baseline, target, dist=dist, seed=seed + 50000 + index)
        if calibration is None:
            calibration_rows.append({'model': model_id, 'n': baseline.n, 'ess': float(baseline.n), 'ess_fraction': 1.0, 'entropy_ess': float(baseline.n), 'max_weight': float(np.max(ratio_weights)), 'weight_cv': 0.0, 'log_normalizer': 0.0, 'target_mass_within_baseline_support': np.nan, 'status': ratio_note})
        else:
            support_mass = np.nan
            if model_id in {'model1', 'model3a', 'model3b'}:
                eta_values = _eta_values(baseline)
                support_mass = target.probability_between(float(np.min(eta_values)), float(np.max(eta_values)))
            calibration_rows.append({'model': model_id, **calibration.diagnostics.as_record(), 'target_mass_within_baseline_support': support_mass, 'status': ratio_note})
        _save_stage_npz(outdir / 'samples' / f'{model_id}_baseline.npz', baseline, {'density_ratio_diagnostic': ratio_weights})
        _save_stage_npz(outdir / 'samples' / f'{model_id}_observation_informed.npz', observation, {'observation_informed': w_obs})
        if model_id in {'model1', 'model3a'}:
            plot_marginal_calibration(baseline, observation, target, outdir / 'figures' / f'{model_id}_eta_hz_calibration.pdf')
            plot_prior_posterior_heatmaps(baseline, observation, outdir / 'figures' / f'{model_id}_prior_posterior_heatmaps.pdf')
    pd.DataFrame(summary_rows).to_csv(outdir / 'prior_posterior_summary.csv', index=False)
    pd.DataFrame(calibration_rows).to_csv(outdir / 'proposal_overlap_diagnostics.csv', index=False)
    pd.DataFrame(regime_rows).to_csv(outdir / 'regime_mass_changes.csv', index=False)
    pd.DataFrame(_parameter_support_rows()).to_csv(outdir / 'parameter_support.csv', index=False)
    pd.DataFrame([d.as_record() for d in BRYSON_ETA_HZ_SCENARIOS.values()]).to_csv(outdir / 'observational_constraints.csv', index=False)
    super_rows = []
    super_rows.extend(_assemble_supermodels(outdir, baseline_ensembles, 'baseline'))
    super_rows.extend(_assemble_supermodels(outdir, observation_ensembles, 'observation_informed'))
    pd.DataFrame(super_rows).to_csv(outdir / 'supermodel_prior_posterior_summary.csv', index=False)
    if run_seti_sensitivity:
        logq_grid = np.linspace(seti_logq_min, seti_logq_max, int(seti_grid_points))
        seti_rows = []
        for model_id in MODEL_NAMES:
            observation = observation_ensembles[model_id]
            seti_rows.extend(_run_seti_sensitivity(observation, _base_weights(observation), logq_grid=logq_grid, n_galactic_targets=n_galactic_targets, settlement_bridge=settlement_bridge, allow_model3b=allow_model3b_seti))
        seti_df = pd.DataFrame(seti_rows)
        seti_df.to_csv(outdir / 'seti_detectability_sensitivity.csv', index=False)
        plot_seti_sensitivity(seti_df, outdir / 'figures' / 'seti_ess_sensitivity.pdf')
    sensitivity_rows: list[dict[str, object]] = []
    if sensitivity_samples > 0:
        for index, model_id in enumerate(MODEL_NAMES):
            print(f'[INFO] Prior sensitivity for {model_id} ...')
            sensitivity_rows.extend(_run_prior_sensitivity(model_id=model_id, n=sensitivity_samples, seed=seed + 200000 + 5009 * index, baseline_dist=dist if dist != 'mixed' else 'lognormal'))
        pd.DataFrame(sensitivity_rows).to_csv(outdir / 'prior_sensitivity.csv', index=False)
    metadata = {'samples_per_model': int(samples), 'sampling_distribution': dist, 'seed': int(seed), 'eta_hz_scenario': eta_scenario, 'eta_hz_interpretation': 'external posterior marginal approximation propagated directly; the original Kepler likelihood is not reconstructed', 'density_ratio_role': 'proposal-overlap diagnostic only; direct external-posterior propagation is used for headline observation-informed results', 'seti_survey': MWA_VELA_2020.__dict__, 'seti_interpretation': 'conditional sensitivity over q_eff; not a universal SETI posterior', 'model3b_seti_enabled': bool(allow_model3b_seti), 'model3b_seti_default_reason': 'disabled to avoid double use of embedded silence conditioning'}
    (outdir / 'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'[OK] Comment 2 outputs written to {outdir.resolve()}')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Observation-informed and prior-sensitivity analysis for reviewer Comment 2.')
    parser.add_argument('--out', default='outputs_comment2')
    parser.add_argument('--samples', type=int, default=120000)
    parser.add_argument('--dist', choices=DISTRIBUTIONS, default='lognormal')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--eta-scenario', choices=tuple(BRYSON_ETA_HZ_SCENARIOS), default='conservative_low')
    parser.add_argument('--sensitivity-samples', type=int, default=20000)
    parser.add_argument('--skip-seti-sensitivity', action='store_true')
    parser.add_argument('--seti-logq-min', type=float, default=-14.0)
    parser.add_argument('--seti-logq-max', type=float, default=-2.0)
    parser.add_argument('--seti-grid-points', type=int, default=25)
    parser.add_argument('--n-galactic-targets', type=float, default=100000000000.0)
    parser.add_argument('--settlement-bridge', choices=['site', 'source'], default='site')
    parser.add_argument('--allow-model3b-seti', action='store_true', help='Apply a second SETI likelihood to Model IIIB-M despite its embedded silence conditioning.')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    run_comment2(outdir=Path(args.out), samples=args.samples, dist=args.dist, seed=args.seed, eta_scenario=args.eta_scenario, sensitivity_samples=args.sensitivity_samples, run_seti_sensitivity=not args.skip_seti_sensitivity, seti_logq_min=args.seti_logq_min, seti_logq_max=args.seti_logq_max, seti_grid_points=args.seti_grid_points, n_galactic_targets=args.n_galactic_targets, settlement_bridge=args.settlement_bridge, allow_model3b_seti=args.allow_model3b_seti)
if __name__ == '__main__':
    main()
