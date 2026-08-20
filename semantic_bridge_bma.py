from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
MODEL_ORDER = ('model1', 'model2', 'model3a', 'model3b', 'model4')
MODEL_LABELS = {'model1': 'Model I', 'model2': 'Model II', 'model3a': 'Model IIIA', 'model3b': 'Model IIIB-M', 'model4': 'Model IV'}
SETTLEMENT_MODELS = {'model3a', 'model3b'}
PRIMARY_BMA_MODELS = ('model1', 'model2', 'model3a', 'model4')

@dataclass(frozen=True)
class Ensemble:
    model_id: str
    path: Path
    arrays: dict[str, np.ndarray]
    weights: np.ndarray

    @property
    def n(self) -> int:
        return int(self.arrays['L_active'].size)

def _load_npz(model_id: str, path: Path) -> Ensemble:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    required = {'L_active', 'N_site', 'N_source'}
    missing = sorted(required - arrays.keys())
    if missing:
        raise KeyError(f'{path} is missing required arrays: {missing}')
    n = int(np.asarray(arrays['L_active']).size)
    for key in required:
        if np.asarray(arrays[key]).size != n:
            raise ValueError(f'{path}: array {key!r} has inconsistent length')
    if 'weight_baseline' in arrays:
        weights = np.asarray(arrays['weight_baseline'], dtype=float).reshape(-1)
    elif 'weight' in arrays:
        weights = np.asarray(arrays['weight'], dtype=float).reshape(-1)
    else:
        weights = np.ones(n, dtype=float)
    valid_w = np.isfinite(weights) & (weights >= 0.0)
    weights = np.where(valid_w, weights, 0.0)
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError(f'{path}: non-positive total sample weight')
    weights /= total
    return Ensemble(model_id=model_id, path=path, arrays=arrays, weights=weights)

def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: Iterable[float]) -> np.ndarray:
    x = np.asarray(values, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    q = np.asarray(tuple(quantiles), dtype=float)
    if np.any((q < 0.0) | (q > 1.0)):
        raise ValueError('quantiles must lie in [0, 1]')
    valid = np.isfinite(x) & np.isfinite(w) & (w > 0.0)
    if not np.any(valid):
        raise ValueError('no valid weighted observations')
    x = x[valid]
    w = w[valid]
    order = np.argsort(x)
    x = x[order]
    w = w[order]
    cumulative = np.cumsum(w)
    cumulative /= cumulative[-1]
    return np.interp(q, cumulative, x)

def _semantic_population(ensemble: Ensemble, bridge: str) -> np.ndarray:
    if bridge not in {'site', 'source'}:
        raise ValueError("bridge must be 'site' or 'source'")
    if ensemble.model_id in SETTLEMENT_MODELS:
        key = 'N_site' if bridge == 'site' else 'N_source'
    else:
        key = 'N_civ' if 'N_civ' in ensemble.arrays else 'N_active'
    population = np.asarray(ensemble.arrays[key], dtype=float).reshape(-1)
    if np.any(~np.isfinite(population)) or np.any(population <= 0.0):
        raise ValueError(f'{ensemble.path}: invalid values in {key}')
    return population

def _region_masses(lifetime: np.ndarray, abundance: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    L = np.asarray(lifetime, dtype=float)
    N = np.asarray(abundance, dtype=float)
    w = np.asarray(weights, dtype=float)
    high = (L >= 10000.0) & (N >= 100.0)
    masks = {'short_lived': L < 1000.0, 'medium_lived': (L >= 1000.0) & (L < 100000.0) & ~high, 'long_lived_localized': (L >= 100000.0) & (N < 100.0), 'high_abundance': high}
    return {name: float(np.sum(w[mask])) for name, mask in masks.items()}

def _bridge_summary(label: str, lifetime: np.ndarray, site_abundance: np.ndarray, source_abundance: np.ndarray, weights: np.ndarray) -> dict[str, object]:
    log_site = np.log10(site_abundance)
    log_source = np.log10(source_abundance)
    q_site = _weighted_quantile(log_site, weights, (0.05, 0.5, 0.95))
    q_source = _weighted_quantile(log_source, weights, (0.05, 0.5, 0.95))
    width_site = float(q_site[2] - q_site[0])
    width_source = float(q_source[2] - q_source[0])
    if width_site <= 0.0:
        raise ValueError(f'{label}: non-positive site-bridge W90')
    regions_site = _region_masses(lifetime, site_abundance, weights)
    regions_source = _region_masses(lifetime, source_abundance, weights)
    changes = {name: regions_source[name] - regions_site[name] for name in regions_site}
    return {'entity': label, 'median_log10_site': float(q_site[1]), 'median_log10_source': float(q_source[1]), 'delta_median_site_minus_source_dex': float(q_site[1] - q_source[1]), 'w90_site_dex': width_site, 'w90_source_dex': width_source, 'w90_source_over_site': width_source / width_site, 'max_absolute_region_mass_change': max((abs(value) for value in changes.values())), 'max_absolute_region_mass_change_percentage_points': 100.0 * max((abs(value) for value in changes.values())), **{f'site_{key}': value for key, value in regions_site.items()}, **{f'source_{key}': value for key, value in regions_source.items()}, **{f'source_minus_site_{key}': value for key, value in changes.items()}}

def _semantic_bridge_analysis(ensembles: dict[str, Ensemble]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_id in ('model3a', 'model3b'):
        ensemble = ensembles[model_id]
        rows.append(_bridge_summary(MODEL_LABELS[model_id], np.asarray(ensemble.arrays['L_active'], dtype=float), _semantic_population(ensemble, 'site'), _semantic_population(ensemble, 'source'), ensemble.weights))
    lifetimes: list[np.ndarray] = []
    site_populations: list[np.ndarray] = []
    source_populations: list[np.ndarray] = []
    mixture_weights: list[np.ndarray] = []
    for model_id in MODEL_ORDER:
        ensemble = ensembles[model_id]
        lifetimes.append(np.asarray(ensemble.arrays['L_active'], dtype=float))
        site_populations.append(_semantic_population(ensemble, 'site'))
        source_populations.append(_semantic_population(ensemble, 'source'))
        mixture_weights.append(ensemble.weights / len(MODEL_ORDER))
    combined_weights = np.concatenate(mixture_weights)
    combined_weights /= np.sum(combined_weights)
    rows.append(_bridge_summary('Neutral equal-family reference mixture', np.concatenate(lifetimes), np.concatenate(site_populations), np.concatenate(source_populations), combined_weights))
    return rows

def _logsumexp(values: np.ndarray, axis: int | None=None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    maximum = np.max(values, axis=axis, keepdims=True)
    with np.errstate(under='ignore'):
        result = maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True))
    if axis is not None:
        result = np.squeeze(result, axis=axis)
    return result

def _marginalized_log_likelihood(population: np.ndarray, *, log10_q_min: float, log10_q_max: float, n_targets: int, n_galactic_targets: float, recovery_efficiency: float, quadrature_nodes: int, chunk_size: int) -> np.ndarray:
    if not log10_q_min < log10_q_max <= 0.0:
        raise ValueError('Require log10_q_min < log10_q_max <= 0')
    if quadrature_nodes < 4:
        raise ValueError('quadrature_nodes must be at least 4')
    if n_targets <= 0 or n_galactic_targets <= 0.0:
        raise ValueError('target counts must be positive')
    if not 0.0 <= recovery_efficiency <= 1.0:
        raise ValueError('recovery_efficiency must lie in [0, 1]')
    pop = np.asarray(population, dtype=float).reshape(-1)
    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_nodes))
    midpoint = 0.5 * (log10_q_min + log10_q_max)
    half_width = 0.5 * (log10_q_max - log10_q_min)
    q_eff = 10.0 ** (midpoint + half_width * nodes)
    log_quadrature_weights = np.log(0.5 * weights)
    result = np.empty(pop.size, dtype=float)
    for start in range(0, pop.size, int(chunk_size)):
        stop = min(start + int(chunk_size), pop.size)
        mean_per_target = pop[start:stop, None] * q_eff[None, :] / n_galactic_targets
        host_probability = -np.expm1(-np.clip(mean_per_target, 0.0, 700.0))
        expected_detections = float(n_targets) * recovery_efficiency * host_probability
        log_likelihood = -expected_detections
        result[start:stop] = _logsumexp(log_likelihood + log_quadrature_weights[None, :], axis=1)
    return result

def _log_evidence(ensemble: Ensemble, population: np.ndarray, args: argparse.Namespace) -> float:
    sample_log_likelihood = _marginalized_log_likelihood(population, log10_q_min=args.log10_q_min, log10_q_max=args.log10_q_max, n_targets=args.n_targets, n_galactic_targets=args.n_galactic_targets, recovery_efficiency=args.recovery_efficiency, quadrature_nodes=args.quadrature_nodes, chunk_size=args.chunk_size)
    log_weights = np.log(ensemble.weights)
    return float(np.asarray(_logsumexp(sample_log_likelihood + log_weights)).item())

def _bma_for_model_set(ensembles: dict[str, Ensemble], model_ids: tuple[str, ...], bridge: str, args: argparse.Namespace, analysis_set: str) -> list[dict[str, object]]:
    log_evidences = np.array([_log_evidence(ensembles[model_id], _semantic_population(ensembles[model_id], bridge), args) for model_id in model_ids], dtype=float)
    log_priors = np.full(len(model_ids), -np.log(len(model_ids)), dtype=float)
    log_posterior = log_evidences + log_priors
    log_posterior -= float(np.asarray(_logsumexp(log_posterior)).item())
    posterior = np.exp(log_posterior)
    rows: list[dict[str, object]] = []
    for model_id, log_evidence, weight in zip(model_ids, log_evidences, posterior):
        rows.append({'analysis_set': analysis_set, 'bridge': bridge, 'model': model_id, 'model_label': MODEL_LABELS[model_id], 'model_prior': 1.0 / len(model_ids), 'log_evidence': float(log_evidence), 'evidence': float(np.exp(log_evidence)), 'posterior_model_weight': float(weight)})
    return rows

def _bma_analysis(ensembles: dict[str, Ensemble], args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for bridge in ('site', 'source'):
        rows.extend(_bma_for_model_set(ensembles, PRIMARY_BMA_MODELS, bridge, args, analysis_set='primary_four_model'))
        rows.extend(_bma_for_model_set(ensembles, MODEL_ORDER, bridge, args, analysis_set='all_five_sensitivity'))
    return rows

def _bma_posterior_predictive(ensembles: dict[str, Ensemble], args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for bridge in ('site', 'source'):
        log_evidences: list[float] = []
        log_likelihoods: dict[str, np.ndarray] = {}
        for model_id in PRIMARY_BMA_MODELS:
            ensemble = ensembles[model_id]
            population = _semantic_population(ensemble, bridge)
            log_likelihood = _marginalized_log_likelihood(population, log10_q_min=args.log10_q_min, log10_q_max=args.log10_q_max, n_targets=args.n_targets, n_galactic_targets=args.n_galactic_targets, recovery_efficiency=args.recovery_efficiency, quadrature_nodes=args.quadrature_nodes, chunk_size=args.chunk_size)
            log_likelihoods[model_id] = log_likelihood
            log_evidence = float(np.asarray(_logsumexp(log_likelihood + np.log(ensemble.weights))).item())
            log_evidences.append(log_evidence)
        log_model_posterior = np.asarray(log_evidences) - np.log(len(PRIMARY_BMA_MODELS))
        log_model_posterior -= float(np.asarray(_logsumexp(log_model_posterior)).item())
        model_posterior = np.exp(log_model_posterior)
        lifetimes: list[np.ndarray] = []
        populations: list[np.ndarray] = []
        mixture_weights: list[np.ndarray] = []
        for model_id, model_weight, log_evidence in zip(PRIMARY_BMA_MODELS, model_posterior, log_evidences):
            ensemble = ensembles[model_id]
            within_log_weight = np.log(ensemble.weights) + log_likelihoods[model_id] - log_evidence
            within_weight = np.exp(within_log_weight)
            within_weight /= np.sum(within_weight)
            mixture_weights.append(model_weight * within_weight)
            lifetimes.append(np.asarray(ensemble.arrays['L_active'], dtype=float))
            populations.append(_semantic_population(ensemble, bridge))
        lifetime = np.concatenate(lifetimes)
        population = np.concatenate(populations)
        weights = np.concatenate(mixture_weights)
        weights /= np.sum(weights)
        q_lifetime = _weighted_quantile(np.log10(lifetime), weights, (0.05, 0.5, 0.95))
        q_population = _weighted_quantile(np.log10(population), weights, (0.05, 0.5, 0.95))
        regions = _region_masses(lifetime, population, weights)
        rows.append({'bridge': bridge, 'median_log10_lifetime': float(q_lifetime[1]), 'w90_log10_lifetime': float(q_lifetime[2] - q_lifetime[0]), 'median_log10_abundance': float(q_population[1]), 'w90_log10_abundance': float(q_population[2] - q_population[0]), **regions})
    return rows

def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f'No rows to write to {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def _primary_weight(bma_rows: list[dict[str, object]], bridge: str, model_id: str) -> float:
    matches = [row for row in bma_rows if row['analysis_set'] == 'primary_four_model' and row['bridge'] == bridge and (row['model'] == model_id)]
    if len(matches) != 1:
        raise RuntimeError(f'Could not resolve primary BMA row for {bridge}/{model_id}')
    return float(matches[0]['posterior_model_weight'])

def _write_latex_table(path: Path, semantic_rows: list[dict[str, object]], bma_rows: list[dict[str, object]], predictive_rows: list[dict[str, object]]) -> None:
    semantic_lookup = {str(row['entity']): row for row in semantic_rows}
    table_rows = [('Model~IIIA', semantic_lookup['Model IIIA']), ('Model~IIIB-M', semantic_lookup['Model IIIB-M']), ('Neutral equal-family reference mixture', semantic_lookup['Neutral equal-family reference mixture'])]
    predictive_lookup = {str(row['bridge']): row for row in predictive_rows}
    lines = ['\\begin{table}[H]', '\\color{red}', '\\centering', '\\caption{Semantic-bridge sensitivity and bridge-conditional Bayesian model-averaging (BMA) results obtained by post-processing the stored baseline ensembles. Panel~(a) reports the metrics defined in Equation~\\eqref{eq:semantic-bridge-metrics}. Panel~(b) reports posterior model weights under equal model priors and the MWA Vela zero-detection likelihood marginalized over a log-uniform $q_{\\mathrm{eff}}$ prior on $[-14,-2]$, and panel~(c) summarizes the resulting BMA posterior predictive distribution. Model~IIIB-M is excluded from the primary BMA to avoid applying a second silence likelihood to a model that already contains explicit silence conditioning.}', '\\label{tab:semantic-bridge-bma}', '\\footnotesize', '\\renewcommand{\\arraystretch}{1.12}', '\\setlength{\\tabcolsep}{5pt}', '\\textbf{(a) Semantic-bridge sensitivity}\\par\\smallskip', '\\begin{tabularx}{\\textwidth}{>{\\raggedright\\arraybackslash}Xccc}', '\\toprule', '\\textbf{Entity} & $\\boldsymbol{\\Delta_{\\mathrm{med}}}$ \\textbf{[dex]} & $\\boldsymbol{R_{W90}}$ & $\\boldsymbol{\\Delta P_{\\max}}$ \\textbf{[p.p.]} \\\\', '\\midrule']
    for latex_label, row in table_rows:
        lines.append(f"{latex_label} & {float(row['delta_median_site_minus_source_dex']):.3f} & {float(row['w90_source_over_site']):.3f} & {float(row['max_absolute_region_mass_change_percentage_points']):.1f} \\\\")
    lines.extend(['\\bottomrule', '\\end{tabularx}', '\\par\\medskip', '\\textbf{(b) Bridge-conditional BMA weights}\\par\\smallskip', '\\begin{tabular}{lcc}', '\\toprule', '\\textbf{Model} & $\\boldsymbol{p(M_k\\mid D,B_{\\mathrm{site}})}$ & $\\boldsymbol{p(M_k\\mid D,B_{\\mathrm{source}})}$ \\\\', '\\midrule'])
    for model_id in PRIMARY_BMA_MODELS:
        lines.append(f"{MODEL_LABELS[model_id].replace('Model ', 'Model~')} & {_primary_weight(bma_rows, 'site', model_id):.3f} & {_primary_weight(bma_rows, 'source', model_id):.3f} \\\\")
    lines.extend(['\\bottomrule', '\\end{tabular}', '\\par\\medskip', '\\textbf{(c) BMA posterior predictive summary}\\par\\smallskip', '\\begin{tabular}{lcccc}', '\\toprule', '\\textbf{Bridge} & $\\boldsymbol{\\operatorname{med}(\\log_{10}L)}$ & $\\boldsymbol{\\operatorname{med}(\\log_{10}N^\\ast)}$ & $\\boldsymbol{W_{90,N}}$ \\textbf{[dex]} & $\\boldsymbol{P(\\mathcal{R}_{\\mathrm{high}})}$ \\textbf{[\\%]} \\\\', '\\midrule'])
    for bridge, latex_label in (('site', '$B_{\\mathrm{site}}$'), ('source', '$B_{\\mathrm{source}}$')):
        row = predictive_lookup[bridge]
        lines.append(f"{latex_label} & {float(row['median_log10_lifetime']):.3f} & {float(row['median_log10_abundance']):.3f} & {float(row['w90_log10_abundance']):.3f} & {100.0 * float(row['high_abundance']):.1f} \\\\")
    lines.extend(['\\bottomrule', '\\end{tabular}', '\\end{table}', ''])
    path.write_text('\n'.join(lines), encoding='utf-8')

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-dir', type=Path)
    parser.add_argument('--model1', type=Path)
    parser.add_argument('--model2', type=Path)
    parser.add_argument('--model3a', type=Path)
    parser.add_argument('--model3b', type=Path)
    parser.add_argument('--model4', type=Path)
    parser.add_argument('--out', '--out-dir', dest='out', type=Path, default=Path('outputs_comment3'))
    parser.add_argument('--log10-q-min', type=float, default=-14.0)
    parser.add_argument('--log10-q-max', type=float, default=-2.0)
    parser.add_argument('--n-targets', type=int, default=10000000)
    parser.add_argument('--n-galactic-targets', type=float, default=100000000000.0)
    parser.add_argument('--recovery-efficiency', type=float, default=1.0)
    parser.add_argument('--quadrature-nodes', type=int, default=48)
    parser.add_argument('--chunk-size', type=int, default=20000)
    return parser.parse_args()

def main() -> None:
    args = _parse_args()
    if args.sample_dir is not None:
        paths = {model_id: args.sample_dir / f'{model_id}.npz' for model_id in MODEL_ORDER}
    else:
        values = {model_id: getattr(args, model_id) for model_id in MODEL_ORDER}
        missing = [model_id for model_id, value in values.items() if value is None]
        if missing:
            raise ValueError(f'Missing model paths: {missing}')
        paths = {model_id: Path(value) for model_id, value in values.items()}
    ensembles = {model_id: _load_npz(model_id, path) for model_id, path in paths.items()}
    semantic_rows = _semantic_bridge_analysis(ensembles)
    bma_rows = _bma_analysis(ensembles, args)
    predictive_rows = _bma_posterior_predictive(ensembles, args)
    args.out.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out / 'semantic_bridge_sensitivity.csv', semantic_rows)
    _write_csv(args.out / 'bma_weights.csv', bma_rows)
    _write_csv(args.out / 'bma_posterior_predictive.csv', predictive_rows)
    _write_latex_table(args.out / 'comment3_table.tex', semantic_rows, bma_rows, predictive_rows)
    metadata = {'input_files': {key: str(value.resolve()) for key, value in paths.items()}, 'samples_per_model': {key: ensembles[key].n for key in MODEL_ORDER}, 'bridges': ['site', 'source'], 'primary_bma_models': list(PRIMARY_BMA_MODELS), 'primary_bma_prior': 'equal', 'model3b_primary_bma': 'excluded_to_avoid_second_silence_likelihood', 'seti_survey': 'MWA Vela 2020', 'n_targets': args.n_targets, 'n_detections': 0, 'log10_q_eff_prior': [args.log10_q_min, args.log10_q_max], 'q_eff_prior_family': 'log_uniform', 'n_galactic_targets': args.n_galactic_targets, 'recovery_efficiency': args.recovery_efficiency, 'quadrature_nodes': args.quadrature_nodes, 'note': 'All results are post-processing of stored ensembles; no model equations were rerun.'}
    (args.out / 'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'[OK] Wrote Comment 3 results to {args.out.resolve()}')
if __name__ == '__main__':
    main()
