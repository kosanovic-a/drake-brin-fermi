from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

def safe_log10(x, floor: float=1e-300) -> np.ndarray:
    return np.log10(np.maximum(np.asarray(x, dtype=float), floor))

def normalized_weights(w, n: int | None=None) -> np.ndarray:
    if w is None:
        if n is None:
            raise ValueError('n is required when w is None')
        return np.full(int(n), 1.0 / int(n))
    w = np.asarray(w, dtype=float).reshape(-1)
    ok = np.isfinite(w) & (w >= 0.0)
    w = np.where(ok, w, 0.0)
    total = float(np.sum(w))
    if total <= 0.0:
        return np.full(w.size, 1.0 / w.size)
    return w / total

def weighted_quantile(x, w, qs) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    w = normalized_weights(w, x.size)
    qs = np.asarray(qs, dtype=float)
    ok = np.isfinite(x) & np.isfinite(w)
    x = x[ok]
    w = normalized_weights(w[ok])
    order = np.argsort(x)
    x = x[order]
    w = w[order]
    c = np.cumsum(w)
    return np.interp(qs, c, x)

def weighted_mean(x, w) -> float:
    x = np.asarray(x, dtype=float)
    w = normalized_weights(w, x.size)
    ok = np.isfinite(x) & np.isfinite(w)
    return float(np.sum(x[ok] * normalized_weights(w[ok])))

def weighted_variance(x, w) -> float:
    x = np.asarray(x, dtype=float)
    w = normalized_weights(w, x.size)
    m = weighted_mean(x, w)
    return float(np.sum(w * (x - m) ** 2))

def effective_sample_size(w) -> float:
    w = normalized_weights(w)
    return float(1.0 / np.sum(w * w))

def distribution_distances(x0, w0, x1, w1, bins: int=160) -> dict[str, float]:
    x0 = np.asarray(x0, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    w0 = normalized_weights(w0, x0.size)
    w1 = normalized_weights(w1, x1.size)
    lo = float(min(np.nanquantile(x0, 0.001), np.nanquantile(x1, 0.001)))
    hi = float(max(np.nanquantile(x0, 0.999), np.nanquantile(x1, 0.999)))
    edges = np.linspace(lo, hi, int(bins) + 1)
    p, _ = np.histogram(x0, bins=edges, weights=w0)
    q, _ = np.histogram(x1, bins=edges, weights=w1)
    eps = 1e-15
    p = (p + eps) / np.sum(p + eps)
    q = (q + eps) / np.sum(q + eps)
    return {'wasserstein': float(wasserstein_distance(x0, x1, u_weights=w0, v_weights=w1)), 'jensen_shannon': float(jensenshannon(p, q, base=2.0) ** 2)}

def diagnostic_regime_masks(logL, logN, thresholds=(3.0, 4.0, 5.0, 2.0)) -> dict[str, np.ndarray]:
    short_t, multi_l_t, long_t, multi_n_t = map(float, thresholds)
    logL = np.asarray(logL, dtype=float)
    logN = np.asarray(logN, dtype=float)
    multi = (logL >= multi_l_t) & (logN >= multi_n_t)
    short = logL < short_t
    medium = (logL >= short_t) & (logL < long_t) & ~multi
    long_localized = (logL >= long_t) & (logN < multi_n_t)
    return {'short': short, 'medium': medium, 'long_localized': long_localized, 'multi_site': multi}

def diagnostic_regime_masses(logL, logN, w, thresholds=(3.0, 4.0, 5.0, 2.0)) -> dict[str, float]:
    w = normalized_weights(w, len(logL))
    masks = diagnostic_regime_masks(logL, logN, thresholds)
    return {name: float(np.sum(w[mask])) for name, mask in masks.items()}

def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}

def save_records(records: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(path, index=False)
    return path
