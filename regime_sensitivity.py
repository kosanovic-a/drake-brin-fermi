from __future__ import annotations

from itertools import product
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from figure_style import configure_matplotlib_for_latex, savefig_publication
configure_matplotlib_for_latex()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis_utils import diagnostic_regime_masses, normalized_weights, safe_log10
from cluster_validation import REGIME_NAMES, threshold_regime_labels
from supermodels import SUPERMODEL_NAMES
from triads import MODEL_NAMES
THRESHOLD_NAMES = ('short_L', 'multi_L', 'long_L', 'multi_N')
BASE_THRESHOLDS = np.array([3.0, 4.0, 5.0, 2.0], dtype=float)

def _load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        log_lifetime = safe_log10(data['L_active'])
        abundance_key = 'N_active' if 'N_active' in data.files else 'N_site'
        log_abundance = safe_log10(data[abundance_key])
        weight = normalized_weights(data['weight'], log_lifetime.size)
    mask = np.isfinite(log_lifetime) & np.isfinite(log_abundance) & np.isfinite(weight)
    return (log_lifetime[mask], log_abundance[mask], normalized_weights(weight[mask]))

def _switch_fraction(base_labels: np.ndarray, shifted_labels: np.ndarray, weight: np.ndarray) -> float:
    return float(np.sum(normalized_weights(weight) * (base_labels != shifted_labels)))

def _record(model: str, log_lifetime: np.ndarray, log_abundance: np.ndarray, weight: np.ndarray, thresholds: np.ndarray, mode: str, changed_threshold: str, delta: float, base_masses: dict[str, float], base_labels: np.ndarray) -> dict[str, float | str]:
    masses = diagnostic_regime_masses(log_lifetime, log_abundance, weight, thresholds)
    labels = threshold_regime_labels(log_lifetime, log_abundance, thresholds)
    row: dict[str, float | str] = {'model': model, 'mode': mode, 'changed_threshold': changed_threshold, 'delta': float(delta), 'short_L': float(thresholds[0]), 'multi_L': float(thresholds[1]), 'long_L': float(thresholds[2]), 'multi_N': float(thresholds[3]), 'classification_switch_fraction': _switch_fraction(base_labels, labels, weight)}
    for name in REGIME_NAMES:
        row[f'P_{name}'] = float(masses[name])
        row[f'delta_P_{name}'] = float(masses[name] - base_masses[name])
        row[f'abs_delta_P_{name}'] = float(abs(masses[name] - base_masses[name]))
    row['max_abs_regime_mass_change'] = float(max((row[f'abs_delta_P_{name}'] for name in REGIME_NAMES)))
    return row

def run_regime_sensitivity(outdir: Path, shifts=(-0.25, 0.0, 0.25)) -> None:
    outdir = Path(outdir)
    target = outdir / 'regime_sensitivity'
    target.mkdir(parents=True, exist_ok=True)
    factorial_records = []
    one_at_a_time_records = []
    for folder, names in (('samples', MODEL_NAMES), ('super_samples', SUPERMODEL_NAMES)):
        for model in names:
            path = outdir / folder / f'{model}.npz'
            if not path.exists():
                continue
            log_lifetime, log_abundance, weight = _load(path)
            base_masses = diagnostic_regime_masses(log_lifetime, log_abundance, weight, BASE_THRESHOLDS)
            base_labels = threshold_regime_labels(log_lifetime, log_abundance, BASE_THRESHOLDS)
            one_at_a_time_records.append(_record(model, log_lifetime, log_abundance, weight, BASE_THRESHOLDS, 'baseline', 'none', 0.0, base_masses, base_labels))
            for threshold_index, threshold_name in enumerate(THRESHOLD_NAMES):
                for delta in shifts:
                    if float(delta) == 0.0:
                        continue
                    thresholds = BASE_THRESHOLDS.copy()
                    thresholds[threshold_index] += float(delta)
                    if thresholds[0] < thresholds[1] < thresholds[2]:
                        one_at_a_time_records.append(_record(model, log_lifetime, log_abundance, weight, thresholds, 'one_at_a_time', threshold_name, float(delta), base_masses, base_labels))
            for delta_tuple in product(shifts, repeat=4):
                thresholds = BASE_THRESHOLDS + np.asarray(delta_tuple, dtype=float)
                if not thresholds[0] < thresholds[1] < thresholds[2]:
                    continue
                record = _record(model, log_lifetime, log_abundance, weight, thresholds, 'full_factorial', 'multiple', float('nan'), base_masses, base_labels)
                for name, value in zip(THRESHOLD_NAMES, delta_tuple):
                    record[f'delta_{name}'] = float(value)
                factorial_records.append(record)
    one_at_a_time = pd.DataFrame(one_at_a_time_records)
    factorial = pd.DataFrame(factorial_records)
    one_at_a_time.to_csv(target / 'regime_one_at_a_time_sensitivity.csv', index=False)
    factorial.to_csv(target / 'regime_threshold_full_factorial.csv', index=False)
    if factorial.empty:
        pd.DataFrame().to_csv(target / 'regime_sensitivity_summary.csv', index=False)
        return
    summary_rows = []
    for model, group in factorial.groupby('model'):
        row = {'model': model, 'max_classification_switch_fraction': float(group['classification_switch_fraction'].max()), 'median_classification_switch_fraction': float(group['classification_switch_fraction'].median()), 'max_abs_regime_mass_change': float(group['max_abs_regime_mass_change'].max())}
        for name in REGIME_NAMES:
            row[f'max_abs_delta_P_{name}'] = float(group[f'abs_delta_P_{name}'].max())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(target / 'regime_sensitivity_summary.csv', index=False)
    fig, axis = plt.subplots(figsize=(8.8, 4.6))
    models = summary['model'].tolist()
    x = np.arange(len(models))
    axis.bar(x, summary['max_classification_switch_fraction'])
    axis.set_xticks(x)
    axis.set_xticklabels(models, rotation=45, ha='right')
    axis.set_ylabel('maximum reclassification fraction')
    axis.grid(True, axis='y', alpha=0.22)
    fig.tight_layout()
    savefig_publication(fig, target / 'regime_reclassification_sensitivity.pdf', dpi=300)
    plt.close(fig)
    sm5 = one_at_a_time[(one_at_a_time['model'] == 'SM5') & (one_at_a_time['mode'] == 'one_at_a_time')]
    if not sm5.empty:
        fig, axis = plt.subplots(figsize=(7.4, 4.4))
        labels = [f'{row.changed_threshold} {row.delta:+.2f}' for row in sm5.itertuples(index=False)]
        x = np.arange(len(sm5))
        axis.bar(x, sm5['classification_switch_fraction'])
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=45, ha='right')
        axis.set_ylabel('reclassification fraction')
        axis.grid(True, axis='y', alpha=0.22)
        fig.tight_layout()
        savefig_publication(fig, target / 'SM5_threshold_one_at_a_time.pdf', dpi=300)
        plt.close(fig)
