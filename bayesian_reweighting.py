from __future__ import annotations

'Numerically stable importance reweighting and weighted summaries.'
from dataclasses import dataclass
from typing import Protocol
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import logsumexp

class Log10Density(Protocol):

    def logpdf_log10(self, log10_x: np.ndarray | float) -> np.ndarray:
        ...

@dataclass(frozen=True)
class WeightDiagnostics:
    n: int
    ess: float
    ess_fraction: float
    entropy_ess: float
    max_weight: float
    coefficient_of_variation: float
    log_normalizer: float

    def as_record(self) -> dict[str, float | int]:
        return {'n': self.n, 'ess': self.ess, 'ess_fraction': self.ess_fraction, 'entropy_ess': self.entropy_ess, 'max_weight': self.max_weight, 'weight_cv': self.coefficient_of_variation, 'log_normalizer': self.log_normalizer}

def _validate_base_weights(base_weights: np.ndarray | None, n: int) -> np.ndarray:
    if base_weights is None:
        return np.full(n, 1.0 / n, dtype=float)
    w = np.asarray(base_weights, dtype=float).reshape(-1)
    if w.size != n:
        raise ValueError(f'Expected {n} base weights, got {w.size}.')
    ok = np.isfinite(w) & (w >= 0.0)
    if not np.all(ok):
        raise ValueError('Base weights must be finite and nonnegative.')
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError('Base weights must have positive sum.')
    return w / total

def normalize_log_weights(log_factor: np.ndarray, base_weights: np.ndarray | None=None) -> tuple[np.ndarray, WeightDiagnostics]:
    lf = np.asarray(log_factor, dtype=float).reshape(-1)
    n = lf.size
    if n == 0:
        raise ValueError('Cannot normalize an empty weight vector.')
    base = _validate_base_weights(base_weights, n)
    if np.any(np.isnan(lf)) or np.any(lf == np.inf):
        raise ValueError('log_factor may contain -inf, but not NaN or +inf.')
    log_unnormalized = np.log(base) + lf
    log_z = float(logsumexp(log_unnormalized))
    if not np.isfinite(log_z):
        raise ValueError('All reweighting factors have zero numerical mass.')
    w = np.exp(log_unnormalized - log_z)
    w /= np.sum(w)
    ess = effective_sample_size(w)
    ent_ess = entropy_effective_sample_size(w)
    mean_w = 1.0 / n
    cv = float(np.std(w) / mean_w)
    diag = WeightDiagnostics(n=n, ess=ess, ess_fraction=ess / n, entropy_ess=ent_ess, max_weight=float(np.max(w)), coefficient_of_variation=cv, log_normalizer=log_z)
    return (w, diag)

def effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float).reshape(-1)
    total = float(np.sum(w))
    if total <= 0.0:
        return 0.0
    w = w / total
    return float(1.0 / np.sum(w * w))

def entropy_effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float).reshape(-1)
    total = float(np.sum(w))
    if total <= 0.0:
        return 0.0
    w = w / total
    positive = w > 0.0
    entropy = -float(np.sum(w[positive] * np.log(w[positive])))
    return float(np.exp(entropy))

def weighted_mean(x: np.ndarray, weights: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    ok = np.isfinite(x) & np.isfinite(w) & (w >= 0.0)
    if not np.any(ok):
        return float('nan')
    total = float(np.sum(w[ok]))
    if total <= 0.0:
        return float('nan')
    return float(np.sum(x[ok] * w[ok]) / total)

def weighted_variance(x: np.ndarray, weights: np.ndarray) -> float:
    mu = weighted_mean(x, weights)
    if not np.isfinite(mu):
        return float('nan')
    x = np.asarray(x, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    ok = np.isfinite(x) & np.isfinite(w) & (w >= 0.0)
    total = float(np.sum(w[ok]))
    return float(np.sum(w[ok] * (x[ok] - mu) ** 2) / total)

def weighted_quantile(x: np.ndarray, weights: np.ndarray, quantiles: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    values = np.asarray(x, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    q = np.asarray(quantiles, dtype=float)
    if np.any((q < 0.0) | (q > 1.0)):
        raise ValueError('Quantiles must lie in [0, 1].')
    ok = np.isfinite(values) & np.isfinite(w) & (w >= 0.0)
    values = values[ok]
    w = w[ok]
    if values.size == 0 or float(np.sum(w)) <= 0.0:
        return np.full(q.shape, np.nan)
    order = np.argsort(values)
    values = values[order]
    w = w[order]
    cumulative = np.cumsum(w) - 0.5 * w
    cumulative /= np.sum(w)
    return np.interp(q, cumulative, values, left=values[0], right=values[-1])

def log_mean_exp(log_values: np.ndarray) -> float:
    arr = np.asarray(log_values, dtype=float).reshape(-1)
    if arr.size == 0:
        return float('-inf')
    return float(logsumexp(arr) - np.log(arr.size))

@dataclass(frozen=True)
class HistogramDensity1D:
    centers: np.ndarray
    density: np.ndarray
    lower: float
    upper: float

    def pdf(self, x: np.ndarray | float) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        result = np.interp(values, self.centers, self.density, left=0.0, right=0.0)
        return np.where((values >= self.lower) & (values <= self.upper), result, 0.0)

    def logpdf(self, x: np.ndarray | float, floor: float=1e-300) -> np.ndarray:
        return np.log(np.maximum(self.pdf(x), floor))

def estimate_histogram_density(x: np.ndarray, weights: np.ndarray | None=None, *, bins: int=180, smooth_sigma_bins: float=1.5, range_padding: float=0.03, pseudocount_fraction: float=1e-10) -> HistogramDensity1D:
    values = np.asarray(x, dtype=float).reshape(-1)
    ok = np.isfinite(values)
    if weights is not None:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.size != values.size:
            raise ValueError('weights and x must have the same length.')
        ok &= np.isfinite(w) & (w >= 0.0)
        w = w[ok]
    else:
        w = None
    values = values[ok]
    if values.size < 20:
        raise ValueError('At least 20 finite samples are required for density estimation.')
    lo = float(np.min(values))
    hi = float(np.max(values))
    span = hi - lo
    if not np.isfinite(span) or span <= 0.0:
        raise ValueError('Density estimation requires non-degenerate samples.')
    lo -= range_padding * span
    hi += range_padding * span
    counts, edges = np.histogram(values, bins=int(bins), range=(lo, hi), weights=w)
    counts = np.asarray(counts, dtype=float)
    pseudo = pseudocount_fraction * max(float(np.sum(counts)), 1.0) / counts.size
    counts += pseudo
    if smooth_sigma_bins > 0.0:
        counts = gaussian_filter1d(counts, sigma=float(smooth_sigma_bins), mode='nearest')
    dx = float(edges[1] - edges[0])
    density = counts / (np.sum(counts) * dx)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return HistogramDensity1D(centers=centers, density=density, lower=lo, upper=hi)

@dataclass(frozen=True)
class MarginalCalibrationResult:
    weights: np.ndarray
    diagnostics: WeightDiagnostics
    proposal_density: HistogramDensity1D
    log_density_ratio: np.ndarray
    log10_values: np.ndarray

def calibrate_positive_marginal(values: np.ndarray, target_density: Log10Density, *, base_weights: np.ndarray | None=None, bins: int=180, smooth_sigma_bins: float=1.5) -> MarginalCalibrationResult:
    v = np.asarray(values, dtype=float).reshape(-1)
    if np.any(~np.isfinite(v)) or np.any(v <= 0.0):
        raise ValueError('Marginal calibration requires finite positive values.')
    base = _validate_base_weights(base_weights, v.size)
    logv = np.log10(v)
    proposal = estimate_histogram_density(logv, base, bins=bins, smooth_sigma_bins=smooth_sigma_bins)
    log_q = np.asarray(target_density.logpdf_log10(logv), dtype=float)
    log_p = proposal.logpdf(logv)
    log_ratio = log_q - log_p
    weights, diagnostics = normalize_log_weights(log_ratio, base)
    return MarginalCalibrationResult(weights=weights, diagnostics=diagnostics, proposal_density=proposal, log_density_ratio=log_ratio, log10_values=logv)
