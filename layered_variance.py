from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
MODEL_IDS = ('model1', 'model2', 'model3a', 'model3b', 'model4')
MODEL_LABELS = {'model1': 'Model I', 'model2': 'Model II', 'model3a': 'Model IIIA', 'model3b': 'Model IIIB-M', 'model4': 'Model IV'}
BRIDGES = ('site', 'source')
TARGETS = ('logL', 'logN')
TARGET_LABELS = {'logL': '$\\log_{10}L_{\\mathrm{active}}$', 'logN': '$\\log_{10}N^\\ast$'}
COMPONENT_LABELS = {'parameter': 'Parameter uncertainty', 'model': 'Model structure', 'bridge': 'Semantic bridge', 'interaction': 'Model x bridge'}

def _weighted_moments(x: np.ndarray, w: np.ndarray) -> tuple[float, float, int]:
    x = np.asarray(x, dtype=float).reshape(-1)
    w = np.asarray(w, dtype=float).reshape(-1)
    ok = np.isfinite(x) & (x > 0.0) & np.isfinite(w) & (w >= 0.0)
    x = np.log10(x[ok])
    w = w[ok]
    if x.size == 0:
        raise ValueError('No positive finite samples remain after filtering.')
    if float(w.sum()) <= 0.0:
        w = np.ones_like(x)
    w = w / w.sum()
    mean = float(np.sum(w * x))
    variance = float(np.sum(w * (x - mean) ** 2))
    return (mean, variance, int(x.size))

def _read_cell(path: Path, target: str, bridge: str) -> tuple[float, float, int]:
    with np.load(path, allow_pickle=True) as data:
        if target == 'logL':
            key = 'L_active'
        elif target == 'logN':
            key = 'N_site' if bridge == 'site' else 'N_source'
        else:
            raise ValueError(f'Unknown target: {target}')
        if key not in data.files:
            raise KeyError(f'{path} does not contain {key!r}.')
        x = np.asarray(data[key], dtype=float)
        w = np.asarray(data['weight'], dtype=float) if 'weight' in data.files else np.ones_like(x)
    return _weighted_moments(x, w)

def decompose(sample_dir: Path, target: str) -> tuple[dict[str, float], pd.DataFrame]:
    means = np.zeros((len(MODEL_IDS), len(BRIDGES)), dtype=float)
    variances = np.zeros_like(means)
    counts = np.zeros_like(means, dtype=int)
    records: list[dict[str, object]] = []
    for i, model_id in enumerate(MODEL_IDS):
        path = sample_dir / f'{model_id}.npz'
        if not path.exists():
            raise FileNotFoundError(path)
        for j, bridge in enumerate(BRIDGES):
            mean, variance, count = _read_cell(path, target, bridge)
            means[i, j] = mean
            variances[i, j] = variance
            counts[i, j] = count
            records.append({'target': target, 'model': model_id, 'model_label': MODEL_LABELS[model_id], 'bridge': bridge, 'mean': mean, 'within_variance': variance, 'n_valid': count})
    grand_mean = float(means.mean())
    model_means = means.mean(axis=1)
    bridge_means = means.mean(axis=0)
    v_parameter = float(variances.mean())
    v_model = float(np.mean((model_means - grand_mean) ** 2))
    v_bridge = float(np.mean((bridge_means - grand_mean) ** 2))
    residual = means - model_means[:, None] - bridge_means[None, :] + grand_mean
    v_interaction = float(np.mean(residual ** 2))
    v_total = v_parameter + v_model + v_bridge + v_interaction
    components = {'parameter': v_parameter, 'model': v_model, 'bridge': v_bridge, 'interaction': v_interaction}
    shares = {name: value / v_total for name, value in components.items()}
    reconstruction_error = abs(sum(components.values()) - v_total)
    share_error = abs(sum(shares.values()) - 1.0)
    result: dict[str, float] = {'grand_mean': grand_mean, 'total_variance': v_total, 'reconstruction_error': reconstruction_error, 'share_sum_error': share_error}
    for name in components:
        result[f'variance_{name}'] = components[name]
        result[f'share_{name}'] = shares[name]
    return (result, pd.DataFrame.from_records(records))

def _write_figure(summary: pd.DataFrame, out_path: Path) -> None:
    order = ['parameter', 'model', 'bridge', 'interaction']
    targets = ['logL', 'logN']
    y = np.arange(len(targets), dtype=float)
    left = np.zeros(len(targets), dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    for component in order:
        values = np.array([100.0 * float(summary.loc[summary['target'] == target, f'share_{component}'].iloc[0]) for target in targets])
        bars = ax.barh(y, values, left=left, label=COMPONENT_LABELS[component])
        for bar, value, start in zip(bars, values, left):
            if value >= 4.0:
                ax.text(start + value / 2.0, bar.get_y() + bar.get_height() / 2.0, f'{value:.1f}%', ha='center', va='center', fontsize=8)
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels([TARGET_LABELS[t] for t in targets])
    ax.set_xlim(0.0, 100.0)
    ax.set_xlabel('Share of total variance (%)')
    ax.invert_yaxis()
    ax.legend(ncol=2, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.02))
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close(fig)

def _write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    rows = {row['target']: row for _, row in summary.iterrows()}
    text = f"\\begin{{table}}[H]\n\\centering\n\\caption{{Balanced hierarchical variance decomposition of the two integrated diagnostic outputs. Values are percentages of total variance under equal representation of the five model families and two semantic bridges.}}\n\\label{{tab:layered-variance-decomposition}}\n\\small\n\\begin{{tabular}}{{lrrrr}}\n\\toprule\n\\textbf{{Output}} & \\textbf{{Parameters}} & \\textbf{{Model}} & \\textbf{{Bridge}} & \\textbf{{Model $\\times$ bridge}} \\\\\n\\midrule\n$\\log_{{10}}L_{{\\mathrm{{active}}}}$ & {100 * rows['logL']['share_parameter']:.1f} & {100 * rows['logL']['share_model']:.1f} & {100 * rows['logL']['share_bridge']:.1f} & {100 * rows['logL']['share_interaction']:.1f} \\\\\n$\\log_{{10}}N^\\ast$ & {100 * rows['logN']['share_parameter']:.1f} & {100 * rows['logN']['share_model']:.1f} & {100 * rows['logN']['share_bridge']:.1f} & {100 * rows['logN']['share_interaction']:.1f} \\\\\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    path.write_text(text, encoding='utf-8')

def run(sample_dir: Path, out_dir: Path) -> pd.DataFrame:
    summaries: list[dict[str, object]] = []
    cell_frames: list[pd.DataFrame] = []
    for target in TARGETS:
        result, cells = decompose(sample_dir, target)
        summaries.append({'target': target, **result})
        cell_frames.append(cells)
    summary = pd.DataFrame.from_records(summaries)
    cells = pd.concat(cell_frames, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / 'layered_variance_components.csv', index=False)
    cells.to_csv(out_dir / 'layered_variance_cell_moments.csv', index=False)
    _write_figure(summary, out_dir / 'layered_variance_decomposition.pdf')
    _write_latex_table(summary, out_dir / 'layered_variance_table.tex')
    metadata = {'models': list(MODEL_IDS), 'bridges': list(BRIDGES), 'targets': list(TARGETS), 'model_weighting': 'equal', 'bridge_weighting': 'equal', 'variance_definition': 'population variance (ddof=0)', 'sample_dir': str(sample_dir.resolve()), 'checks': {row['target']: {'reconstruction_error': float(row['reconstruction_error']), 'share_sum_error': float(row['share_sum_error'])} for row in summaries}}
    (out_dir / 'layered_variance_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return summary

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Balanced hierarchical variance decomposition for the Drake-Fermi-Brin ensembles.')
    parser.add_argument('--sample-dir', default='outputs/samples')
    parser.add_argument('--out', default='outputs/layered_variance')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    summary = run(Path(args.sample_dir), Path(args.out))
    columns = ['target', 'share_parameter', 'share_model', 'share_bridge', 'share_interaction']
    print(summary[columns].to_string(index=False, float_format=lambda x: f'{100 * x:.3f}%' if x <= 1.0 else f'{x:.6g}'))
    print(f'[OK] outputs: {Path(args.out).resolve()}')
if __name__ == '__main__':
    main()
