from __future__ import annotations

'Prior-sensitivity utilities for bounded parameter triads.'
from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
FRACTION_PARAMETERS = {'fp', 'fl', 'fi', 'fa', 'fg', 'p_set', 'fpm', 'fm', 'fj', 'fme'}
WEAK_PARAMETER_GROUPS: Mapping[str, tuple[str, ...]] = {'model1': ('fl', 'fi', 'fa', 'N_active'), 'model2': ('F_bioactive', 'N_active'), 'model3a': ('fl', 'fi', 'fa', 'B_density', 'fg', 'ne_br', 'p_set', 'v_eff', 'R', 't_delay', 'lambda_loc', 'lambda_corr', 'N50', 'gamma', 'T_max', 'N_gal_cap'), 'model3b': ('fl', 'fi', 'fa', 'M_site', 'N_gal_cap', 'lambda_loc', 'lambda_corr', 'N50', 'gamma', 'T_max', 'T_det', 'T_vis', 'chi_vis', 'T50_silence', 'eta_silence', 'lambda_det', 'lambda_vis'), 'model4': ('fg', 'fpm', 'fm', 'fj', 'fme', 'N_active')}

@dataclass(frozen=True)
class TriadVariant:
    name: str
    width_scale: float = 1.0
    peak_shift_dex: float = 0.0

    def __post_init__(self) -> None:
        if self.width_scale <= 0.0:
            raise ValueError('width_scale must be positive.')
DEFAULT_VARIANTS = (TriadVariant('baseline'), TriadVariant('narrow_075', width_scale=0.75), TriadVariant('wide_125', width_scale=1.25), TriadVariant('peak_low_m025dex', peak_shift_dex=-0.25), TriadVariant('peak_high_p025dex', peak_shift_dex=0.25))

def transform_triad(triad: Sequence[float], parameter_name: str, variant: TriadVariant) -> list[float]:
    lo, peak, hi = map(float, triad)
    if not 0.0 < lo <= peak <= hi:
        raise ValueError(f'Invalid positive triad {triad}.')
    log_lo, log_peak, log_hi = np.log10([lo, peak, hi])
    shifted_peak = log_peak + variant.peak_shift_dex
    new_lo = shifted_peak - variant.width_scale * (log_peak - log_lo)
    new_hi = shifted_peak + variant.width_scale * (log_hi - log_peak)
    lo_new, peak_new, hi_new = 10.0 ** np.array([new_lo, shifted_peak, new_hi])
    if parameter_name in FRACTION_PARAMETERS:
        hi_new = min(1.0, hi_new)
        peak_new = min(hi_new, peak_new)
        lo_new = min(lo_new, peak_new)
        lo_new = max(lo_new, 1e-15)
    peak_new = max(lo_new, min(peak_new, hi_new))
    if lo_new == hi_new:
        raise ValueError(f'Variant {variant.name} collapsed triad {parameter_name}.')
    return [float(lo_new), float(peak_new), float(hi_new)]

def apply_variant(triads: Mapping[str, Sequence[float]], parameter_names: Sequence[str], variant: TriadVariant) -> dict[str, list[float]]:
    selected = set(parameter_names)
    unknown = selected.difference(triads)
    if unknown:
        raise KeyError(f'Unknown triad parameter(s): {sorted(unknown)}')
    out: dict[str, list[float]] = {}
    for name, triad in triads.items():
        if name in selected:
            out[name] = transform_triad(triad, name, variant)
        else:
            out[name] = list(map(float, triad))
    return out

def _hist_probability(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(x, bins=edges)
    p = counts.astype(float) + 1e-12
    return p / np.sum(p)

def distribution_distance_summary(baseline: np.ndarray, alternative: np.ndarray, *, bins: int=160) -> dict[str, float]:
    a = np.asarray(baseline, dtype=float)
    b = np.asarray(alternative, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return {'wasserstein': float('nan'), 'jensen_shannon': float('nan')}
    lo = float(min(np.min(a), np.min(b)))
    hi = float(max(np.max(a), np.max(b)))
    if hi <= lo:
        return {'wasserstein': 0.0, 'jensen_shannon': 0.0}
    edges = np.linspace(lo, hi, int(bins) + 1)
    pa = _hist_probability(a, edges)
    pb = _hist_probability(b, edges)
    return {'wasserstein': float(wasserstein_distance(a, b)), 'jensen_shannon': float(jensenshannon(pa, pb, base=2.0) ** 2)}
