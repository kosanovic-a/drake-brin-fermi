from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from figure_style import configure_matplotlib_for_latex, savefig_publication
configure_matplotlib_for_latex()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, completeness_score, davies_bouldin_score, homogeneity_score, normalized_mutual_info_score, silhouette_score, v_measure_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from analysis_utils import diagnostic_regime_masks, normalized_weights, safe_log10
from supermodels import SUPERMODEL_NAMES
from triads import MODEL_NAMES
REGIME_NAMES = ('short', 'medium', 'long_localized', 'multi_site')
GMM_COVARIANCE_TYPES = ('full', 'tied', 'diag', 'spherical')

def _resample_indices(weight: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    probability = normalized_weights(weight)
    n = probability.size
    size = min(int(size), n)
    uniform = np.allclose(probability, np.full(n, 1.0 / n), rtol=1e-08, atol=1e-12)
    if uniform:
        return rng.choice(n, size=size, replace=False)
    return rng.choice(n, size=size, replace=True, p=probability)

def load_diagnostic_data(path: Path, max_rows: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    with np.load(path, allow_pickle=True) as data:
        lifetime = np.asarray(data['L_active'], dtype=float)
        abundance_key = 'N_active' if 'N_active' in data.files else 'N_site'
        abundance = np.asarray(data[abundance_key], dtype=float)
        source = np.asarray(data['N_source'], dtype=float) if 'N_source' in data.files else abundance.copy()
        cap = np.asarray(data['cap_fraction'], dtype=float) if 'cap_fraction' in data.files else np.zeros_like(lifetime)
        cap = np.nan_to_num(cap, nan=0.0, posinf=1.0, neginf=0.0)
        weight = np.asarray(data['weight'], dtype=float) if 'weight' in data.files else np.ones_like(lifetime)
        component = np.asarray(data['component_model']).astype(str) if 'component_model' in data.files else np.full(lifetime.size, str(np.asarray(data['model_id']).item()))
    mask = np.isfinite(lifetime) & (lifetime > 0.0) & np.isfinite(abundance) & (abundance > 0.0) & np.isfinite(source) & (source > 0.0) & np.isfinite(cap) & np.isfinite(weight) & (weight >= 0.0)
    log_lifetime = safe_log10(lifetime[mask])
    log_abundance = safe_log10(abundance[mask])
    log_source = safe_log10(source[mask])
    cap = cap[mask]
    weight = weight[mask]
    component = component[mask]
    indices = _resample_indices(weight, min(int(max_rows), log_lifetime.size), rng)
    diagnostic = np.column_stack([log_lifetime[indices], log_abundance[indices]])
    cluster_scaler = StandardScaler()
    cluster_features = cluster_scaler.fit_transform(diagnostic)
    extended = np.column_stack([log_lifetime[indices], log_abundance[indices], log_abundance[indices] - log_source[indices], cap[indices]])
    names = np.asarray(['logL', 'logN_active', 'logQ_site', 'cap_fraction'])
    variable = np.std(extended, axis=0) > 1e-12
    if not np.any(variable):
        raise ValueError(f'Invalid visualization features in {path}')
    scaler = StandardScaler()
    standardized = scaler.fit_transform(extended[:, variable])
    pca = PCA(n_components=min(3, standardized.shape[1]), svd_solver='full')
    scores = pca.fit_transform(standardized)
    if np.any(~np.isfinite(scores)):
        raise ValueError(f'Invalid PCA scores in {path}')
    return {'raw': diagnostic, 'extended': extended, 'cluster_features': cluster_features, 'cluster_scaler_mean': cluster_scaler.mean_, 'cluster_scaler_scale': cluster_scaler.scale_, 'standardized': standardized, 'scores': scores, 'component': component[indices], 'feature_names': names[variable], 'scaler_mean': scaler.mean_, 'scaler_scale': scaler.scale_, 'pca_components': pca.components_, 'pca_explained_variance_ratio': pca.explained_variance_ratio_}

def threshold_regime_labels(log_lifetime: np.ndarray, log_abundance: np.ndarray, thresholds=(3.0, 4.0, 5.0, 2.0)) -> np.ndarray:
    masks = diagnostic_regime_masks(log_lifetime, log_abundance, thresholds)
    labels = np.full(len(log_lifetime), -1, dtype=int)
    for index, name in enumerate(REGIME_NAMES):
        labels[masks[name]] = index
    if np.any(labels < 0):
        raise ValueError('Threshold-based regime definitions do not partition the diagnostic space')
    return labels

def _pairwise_mean_ari(labels: list[np.ndarray]) -> float:
    values = [adjusted_rand_score(labels[i], labels[j]) for i in range(len(labels)) for j in range(i + 1, len(labels))]
    return float(np.mean(values)) if values else 1.0

def _subsample_stability(features: np.ndarray, k: int, repeats: int, seed: int, fraction: float=0.8) -> float:
    rng = np.random.default_rng(seed)
    n = features.shape[0]
    size = min(n, max(k * 30, int(round(float(fraction) * n))))
    predictions = []
    for repeat in range(int(repeats)):
        indices = rng.choice(n, size=size, replace=False)
        model = KMeans(n_clusters=int(k), n_init=30, random_state=seed + repeat).fit(features[indices])
        predictions.append(model.predict(features))
    return _pairwise_mean_ari(predictions)

def elbow_k(k_values: np.ndarray, inertia: np.ndarray) -> int:
    k_values = np.asarray(k_values, dtype=float)
    inertia = np.asarray(inertia, dtype=float)
    if k_values.size < 3:
        return int(k_values[np.argmin(inertia)])
    x = (k_values - k_values.min()) / max(float(np.ptp(k_values)), 1e-12)
    y = (inertia - inertia.min()) / max(float(np.ptp(inertia)), 1e-12)
    start = np.array([x[0], y[0]])
    end = np.array([x[-1], y[-1]])
    direction = end - start
    norm = np.linalg.norm(direction)
    if norm <= 1e-12:
        return int(k_values[0])
    points = np.column_stack([x, y])
    offset = points - start
    distance = np.abs(direction[0] * offset[:, 1] - direction[1] * offset[:, 0]) / norm
    return int(k_values[int(np.argmax(distance))])

def _gap_statistics(features: np.ndarray, k_values: list[int], reference_count: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    low = np.min(features, axis=0)
    high = np.max(features, axis=0)
    records = []
    for k in k_values:
        observed = KMeans(n_clusters=int(k), n_init=30, random_state=seed + 97 * k).fit(features).inertia_
        reference_logs = []
        for index in range(int(reference_count)):
            reference = rng.uniform(low, high, size=features.shape)
            inertia = KMeans(n_clusters=int(k), n_init=10, random_state=seed + 10000 * k + index).fit(reference).inertia_
            reference_logs.append(np.log(max(float(inertia), 1e-300)))
        reference_logs = np.asarray(reference_logs, dtype=float)
        gap = float(np.mean(reference_logs) - np.log(max(float(observed), 1e-300)))
        standard = float(np.std(reference_logs, ddof=1) * np.sqrt(1.0 + 1.0 / int(reference_count))) if int(reference_count) > 1 else 0.0
        records.append({'k': int(k), 'gap': gap, 'gap_standard_error': standard})
    frame = pd.DataFrame(records)
    selected = int(frame.loc[frame['gap'].idxmax(), 'k'])
    for index in range(len(frame) - 1):
        current = frame.iloc[index]
        following = frame.iloc[index + 1]
        if current['gap'] >= following['gap'] - following['gap_standard_error']:
            selected = int(current['k'])
            break
    frame['gap_selected_k'] = selected
    return frame

def _select_kmeans_k(frame: pd.DataFrame) -> int:
    maximum = float(frame['silhouette'].max())
    candidates = frame[np.isclose(frame['silhouette'], maximum, rtol=0.0, atol=1e-12)]
    return int(candidates.sort_values(['k', 'davies_bouldin'], ascending=[True, True]).iloc[0]['k'])

def _fit_kmeans_grid(features: np.ndarray, k_values: list[int], seed_count: int, subsamples: int, seed: int) -> pd.DataFrame:
    records = []
    for k in k_values:
        label_sets = []
        silhouettes = []
        calinski = []
        davies = []
        inertias = []
        minimum_fractions = []
        for index in range(int(seed_count)):
            model = KMeans(n_clusters=int(k), n_init=30, random_state=seed + 1009 * k + index)
            labels = model.fit_predict(features)
            label_sets.append(labels)
            silhouettes.append(silhouette_score(features, labels, sample_size=min(5000, len(labels)), random_state=seed + 101 * index + k))
            calinski.append(calinski_harabasz_score(features, labels))
            davies.append(davies_bouldin_score(features, labels))
            inertias.append(float(model.inertia_))
            minimum_fractions.append(float(np.min(np.bincount(labels, minlength=int(k))) / len(labels)))
        records.append({'k': int(k), 'silhouette': float(np.mean(silhouettes)), 'silhouette_sd': float(np.std(silhouettes, ddof=1)) if len(silhouettes) > 1 else 0.0, 'calinski_harabasz': float(np.mean(calinski)), 'davies_bouldin': float(np.mean(davies)), 'inertia': float(np.mean(inertias)), 'inertia_sd': float(np.std(inertias, ddof=1)) if len(inertias) > 1 else 0.0, 'seed_ARI': _pairwise_mean_ari(label_sets), 'subsample_ARI': _subsample_stability(features, int(k), int(subsamples), seed + 10000 + k), 'min_cluster_fraction': float(np.mean(minimum_fractions))})
    frame = pd.DataFrame(records)
    selected = _select_kmeans_k(frame)
    frame['selected_kmeans_k'] = selected
    frame['elbow_k'] = elbow_k(frame['k'].to_numpy(), frame['inertia'].to_numpy())
    frame['silhouette_k'] = int(frame.loc[frame['silhouette'].idxmax(), 'k'])
    frame['calinski_harabasz_k'] = int(frame.loc[frame['calinski_harabasz'].idxmax(), 'k'])
    frame['davies_bouldin_k'] = int(frame.loc[frame['davies_bouldin'].idxmin(), 'k'])
    return frame

def _fit_gmm_grid(features: np.ndarray, k_values: list[int], covariance_types: tuple[str, ...], max_rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sample = features if features.shape[0] <= int(max_rows) else features[rng.choice(features.shape[0], size=int(max_rows), replace=False)]
    records = []
    for covariance_type in covariance_types:
        for k in k_values:
            model = GaussianMixture(n_components=int(k), covariance_type=str(covariance_type), n_init=5, reg_covar=1e-06, random_state=seed + 101 * k + 10007 * covariance_types.index(covariance_type)).fit(sample)
            records.append({'k': int(k), 'covariance_type': str(covariance_type), 'mean_log_likelihood': float(model.score(sample)), 'total_log_likelihood': float(model.score(sample) * sample.shape[0]), 'AIC': float(model.aic(sample)), 'BIC': float(model.bic(sample)), 'converged': bool(model.converged_), 'n_iter': int(model.n_iter_), 'n_rows': int(sample.shape[0])})
    frame = pd.DataFrame(records)
    best_index = frame['BIC'].idxmin()
    frame['selected_gmm_k'] = int(frame.loc[best_index, 'k'])
    frame['selected_gmm_covariance'] = str(frame.loc[best_index, 'covariance_type'])
    return frame

def _cluster_regime_comparison(raw: np.ndarray, cluster_labels: np.ndarray, target: Path, dataset: str) -> dict[str, float]:
    regime_labels = threshold_regime_labels(raw[:, 0], raw[:, 1])
    metrics = {'threshold_cluster_ARI': float(adjusted_rand_score(regime_labels, cluster_labels)), 'threshold_cluster_NMI': float(normalized_mutual_info_score(regime_labels, cluster_labels)), 'threshold_cluster_homogeneity': float(homogeneity_score(regime_labels, cluster_labels)), 'threshold_cluster_completeness': float(completeness_score(regime_labels, cluster_labels)), 'threshold_cluster_v_measure': float(v_measure_score(regime_labels, cluster_labels))}
    contingency = pd.crosstab(pd.Categorical([REGIME_NAMES[index] for index in regime_labels], categories=REGIME_NAMES, ordered=True), cluster_labels + 1, rownames=['threshold_regime'], colnames=['cluster'], dropna=False)
    contingency.to_csv(target / f'{dataset}_threshold_cluster_contingency.csv')
    row_normalized = contingency.div(contingency.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    row_normalized.to_csv(target / f'{dataset}_threshold_cluster_contingency_row_fraction.csv')
    fig, axis = plt.subplots(figsize=(5.8, 3.8))
    image = axis.imshow(row_normalized.to_numpy(), aspect='auto', vmin=0.0, vmax=1.0)
    axis.set_xticks(np.arange(row_normalized.shape[1]))
    axis.set_xticklabels([str(value) for value in row_normalized.columns])
    axis.set_yticks(np.arange(len(REGIME_NAMES)))
    axis.set_yticklabels([name.replace('_', ' ') for name in REGIME_NAMES])
    axis.set_xlabel('data-driven cluster')
    axis.set_ylabel('threshold-based regime')
    for row in range(row_normalized.shape[0]):
        for column in range(row_normalized.shape[1]):
            axis.text(column, row, f'{row_normalized.iloc[row, column]:.2f}', ha='center', va='center', fontsize=7)
    fig.colorbar(image, ax=axis, label='row fraction')
    fig.tight_layout()
    savefig_publication(fig, target / f'{dataset}_threshold_cluster_contingency.pdf', dpi=300)
    plt.close(fig)
    return metrics

def _cluster_summary(raw: np.ndarray, labels: np.ndarray, dataset: str) -> pd.DataFrame:
    rows = []
    for cluster in sorted(np.unique(labels)):
        mask = labels == cluster
        rows.append({'dataset': dataset, 'cluster': int(cluster + 1), 'n': int(np.sum(mask)), 'share': float(np.mean(mask)), 'logL_mean': float(np.mean(raw[mask, 0])), 'logL_q05': float(np.quantile(raw[mask, 0], 0.05)), 'logL_q50': float(np.quantile(raw[mask, 0], 0.5)), 'logL_q95': float(np.quantile(raw[mask, 0], 0.95)), 'logN_mean': float(np.mean(raw[mask, 1])), 'logN_q05': float(np.quantile(raw[mask, 1], 0.05)), 'logN_q50': float(np.quantile(raw[mask, 1], 0.5)), 'logN_q95': float(np.quantile(raw[mask, 1], 0.95))})
    return pd.DataFrame(rows)

def _plot_curve(frame: pd.DataFrame, column: str, ylabel: str, path: Path, selected: int, secondary: tuple[str, str] | None=None) -> None:
    fig, axis = plt.subplots(figsize=(5.4, 3.8))
    axis.plot(frame['k'], frame[column], marker='o', label=ylabel)
    if secondary is not None:
        axis.plot(frame['k'], frame[secondary[0]], marker='s', label=secondary[1])
        axis.legend(frameon=False)
    axis.axvline(int(selected), linestyle='--', linewidth=0.9)
    axis.set_xlabel('$K$')
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.22)
    fig.tight_layout()
    savefig_publication(fig, path, dpi=300)
    plt.close(fig)

def _plot_gap(frame: pd.DataFrame, path: Path) -> None:
    selected = int(frame['gap_selected_k'].iloc[0])
    fig, axis = plt.subplots(figsize=(5.4, 3.8))
    axis.errorbar(frame['k'], frame['gap'], yerr=frame['gap_standard_error'], marker='o', capsize=3)
    axis.axvline(selected, linestyle='--', linewidth=0.9)
    axis.set_xlabel('$K$')
    axis.set_ylabel('gap statistic')
    axis.grid(True, alpha=0.22)
    fig.tight_layout()
    savefig_publication(fig, path, dpi=300)
    plt.close(fig)

def _plot_gmm(frame: pd.DataFrame, path: Path) -> None:
    best_by_k = frame.loc[frame.groupby('k')['BIC'].idxmin()].sort_values('k')
    delta_aic = best_by_k['AIC'] - best_by_k['AIC'].min()
    delta_bic = best_by_k['BIC'] - best_by_k['BIC'].min()
    selected = int(frame['selected_gmm_k'].iloc[0])
    fig, axis = plt.subplots(figsize=(5.4, 3.8))
    axis.plot(best_by_k['k'], delta_aic, marker='o', label='$\\Delta$AIC')
    axis.plot(best_by_k['k'], delta_bic, marker='s', label='$\\Delta$BIC')
    axis.axvline(selected, linestyle='--', linewidth=0.9)
    axis.set_xlabel('$K$')
    axis.set_ylabel('information-criterion difference')
    axis.legend(frameon=False)
    axis.grid(True, alpha=0.22)
    fig.tight_layout()
    savefig_publication(fig, path, dpi=300)
    plt.close(fig)

def _plot_clusters(raw: np.ndarray, labels: np.ndarray, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(5.4, 4.2))
    for cluster in sorted(np.unique(labels)):
        mask = labels == cluster
        axis.scatter(raw[mask, 0], raw[mask, 1], s=6, alpha=0.38, edgecolors='none', label=str(cluster + 1), rasterized=True)
    x_low, x_high = axis.get_xlim()
    y_low, y_high = axis.get_ylim()
    axis.plot([3.0, 3.0], [y_low, y_high], linestyle=':', linewidth=0.8)
    axis.plot([5.0, 5.0], [y_low, min(2.0, y_high)], linestyle=':', linewidth=0.8)
    axis.plot([4.0, 4.0], [max(2.0, y_low), y_high], linestyle=':', linewidth=0.8)
    axis.plot([max(4.0, x_low), x_high], [2.0, 2.0], linestyle=':', linewidth=0.8)
    axis.set_xlim(x_low, x_high)
    axis.set_ylim(y_low, y_high)
    axis.set_xlabel('$\\log_{10}(L_{\\mathrm{active}}/\\mathrm{yr})$')
    axis.set_ylabel('$\\log_{10}N_{\\mathrm{plot}}$')
    axis.legend(frameon=False, title='cluster', ncol=2)
    axis.grid(True, alpha=0.18)
    fig.tight_layout()
    savefig_publication(fig, path, dpi=350)
    plt.close(fig)

def validate_dataset(name: str, data: dict[str, np.ndarray], target: Path, k_min: int, k_max: int, seeds: int, subsamples: int, gap_references: int, gmm_rows: int, seed: int) -> dict[str, object]:
    raw = data['raw']
    features = data['cluster_features']
    k_values = list(range(int(k_min), min(int(k_max), features.shape[0] - 1) + 1))
    if len(k_values) == 0:
        raise ValueError(f'Insufficient rows for clustering dataset {name}')
    kmeans = _fit_kmeans_grid(features, k_values, int(seeds), int(subsamples), seed)
    gap = _gap_statistics(features, k_values, int(gap_references), seed + 300000)
    kmeans = kmeans.merge(gap, on='k', how='left')
    gmm = _fit_gmm_grid(features, k_values, GMM_COVARIANCE_TYPES, int(gmm_rows), seed + 600000)
    selected_kmeans = int(kmeans['selected_kmeans_k'].iloc[0])
    final_model = KMeans(n_clusters=selected_kmeans, n_init=100, random_state=seed + 900000)
    labels = final_model.fit_predict(features)
    comparison = _cluster_regime_comparison(raw, labels, target, name)
    summary = _cluster_summary(raw, labels, name)
    summary.to_csv(target / f'{name}_cluster_summary.csv', index=False)
    components = pd.crosstab(labels + 1, data['component'], rownames=['cluster'], colnames=['component'])
    components.to_csv(target / f'{name}_cluster_component_counts.csv')
    components.div(components.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0).to_csv(target / f'{name}_cluster_component_fractions.csv')
    kmeans.insert(0, 'dataset', name)
    gmm.insert(0, 'dataset', name)
    kmeans.to_csv(target / f'{name}_kmeans_validation.csv', index=False)
    gmm.to_csv(target / f'{name}_gmm_validation.csv', index=False)
    _plot_curve(kmeans, 'inertia', 'within-cluster sum of squares', target / f'{name}_kmeans_elbow.pdf', int(kmeans['elbow_k'].iloc[0]))
    _plot_curve(kmeans, 'silhouette', 'silhouette score', target / f'{name}_kmeans_silhouette.pdf', selected_kmeans)
    _plot_curve(kmeans, 'subsample_ARI', 'adjusted Rand index', target / f'{name}_kmeans_stability.pdf', selected_kmeans, secondary=('seed_ARI', 'initialization stability'))
    _plot_gap(kmeans, target / f'{name}_gap_statistic.pdf')
    _plot_gmm(gmm, target / f'{name}_gmm_information_criteria.pdf')
    _plot_clusters(raw, labels, target / f'{name}_data_driven_clusters.pdf')
    gmm_best = gmm.loc[gmm['BIC'].idxmin()]
    selected_row = kmeans[kmeans['k'] == selected_kmeans].iloc[0]
    return {'dataset': name, 'n_rows': int(features.shape[0]), 'n_cluster_features': int(features.shape[1]), 'n_pca_components': int(data['scores'].shape[1]), 'pca_explained_variance': float(np.sum(data['pca_explained_variance_ratio'])), 'selected_kmeans_k': selected_kmeans, 'elbow_k': int(kmeans['elbow_k'].iloc[0]), 'silhouette_k': int(kmeans['silhouette_k'].iloc[0]), 'calinski_harabasz_k': int(kmeans['calinski_harabasz_k'].iloc[0]), 'davies_bouldin_k': int(kmeans['davies_bouldin_k'].iloc[0]), 'gap_k': int(kmeans['gap_selected_k'].iloc[0]), 'selected_silhouette': float(selected_row['silhouette']), 'selected_seed_ARI': float(selected_row['seed_ARI']), 'selected_subsample_ARI': float(selected_row['subsample_ARI']), 'selected_min_cluster_fraction': float(selected_row['min_cluster_fraction']), 'selected_gmm_k': int(gmm_best['k']), 'selected_gmm_covariance': str(gmm_best['covariance_type']), 'selected_gmm_total_log_likelihood': float(gmm_best['total_log_likelihood']), 'selected_gmm_AIC': float(gmm_best['AIC']), 'selected_gmm_BIC': float(gmm_best['BIC']), **comparison}

def run_cluster_validation(outdir: Path, max_rows: int=30000, k_min: int=2, k_max: int=10, seeds: int=8, subsamples: int=12, seed: int=12345, gap_references: int=10, gmm_rows: int=12000) -> None:
    outdir = Path(outdir)
    target = outdir / 'clustering'
    target.mkdir(parents=True, exist_ok=True)
    datasets = [(model, outdir / 'samples' / f'{model}.npz') for model in MODEL_NAMES]
    datasets += [(model, outdir / 'super_samples' / f'{model}.npz') for model in SUPERMODEL_NAMES]
    rows = []
    for index, (name, path) in enumerate(datasets):
        if not path.exists():
            continue
        data = load_diagnostic_data(path, int(max_rows), seed + 1000 * index)
        rows.append(validate_dataset(name, data, target, int(k_min), int(k_max), int(seeds), int(subsamples), int(gap_references), int(gmm_rows), seed + 10000 * index))
    summary = pd.DataFrame(rows)
    summary.to_csv(target / 'cluster_validation_summary.csv', index=False)
    summary.to_csv(target / 'selected_clusters.csv', index=False)
