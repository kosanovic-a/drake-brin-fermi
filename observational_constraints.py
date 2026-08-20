from __future__ import annotations

'Observation-informed densities used by the Comment 2 analysis.\n\nThe numerical values in :data:`BRYSON_ETA_HZ_SCENARIOS` are taken from the\nabstract of Bryson et al. (2021).  The paper reports two estimates for each\nhabitable-zone definition because it uses two limiting extrapolations of the\nKepler completeness beyond the directly calibrated period range.\n\nImportant statistical qualification\n-----------------------------------\nThe published quantities are posterior occurrence-rate summaries, not raw\nlikelihood functions.  The code therefore treats them as *external posterior\nmarginals* for eta_HZ and performs marginal calibration of the baseline Monte\nCarlo ensemble.  It does not claim to reconstruct the original Kepler\nlikelihood or the prior used by Bryson et al.\n'
from dataclasses import dataclass
from typing import Mapping
import numpy as np
from scipy.special import ndtr
_LOG_SQRT_2_OVER_PI = 0.5 * np.log(2.0 / np.pi)

@dataclass(frozen=True)
class AsymmetricLog10Density:
    center: float
    lower_68: float
    upper_68: float
    label: str
    source: str

    def __post_init__(self) -> None:
        if not 0.0 < self.lower_68 < self.center < self.upper_68:
            raise ValueError(f'Expected 0 < lower_68 < center < upper_68, got {self.lower_68}, {self.center}, {self.upper_68}.')

    @property
    def mu(self) -> float:
        return float(np.log10(self.center))

    @property
    def sigma_left(self) -> float:
        return float(self.mu - np.log10(self.lower_68))

    @property
    def sigma_right(self) -> float:
        return float(np.log10(self.upper_68) - self.mu)

    def logpdf_log10(self, log10_x: np.ndarray | float) -> np.ndarray:
        x = np.asarray(log10_x, dtype=float)
        sig_left = self.sigma_left
        sig_right = self.sigma_right
        sigma = np.where(x <= self.mu, sig_left, sig_right)
        log_norm = _LOG_SQRT_2_OVER_PI - np.log(sig_left + sig_right)
        return log_norm - 0.5 * ((x - self.mu) / sigma) ** 2

    def pdf_log10(self, log10_x: np.ndarray | float) -> np.ndarray:
        return np.exp(self.logpdf_log10(log10_x))

    def cdf_log10(self, log10_x: np.ndarray | float) -> np.ndarray:
        x = np.asarray(log10_x, dtype=float)
        sl = self.sigma_left
        sr = self.sigma_right
        denom = sl + sr
        left = 2.0 * sl / denom * ndtr((x - self.mu) / sl)
        right = sl / denom + 2.0 * sr / denom * (ndtr((x - self.mu) / sr) - 0.5)
        return np.where(x <= self.mu, left, right)

    def probability_between(self, lower: float, upper: float) -> float:
        if not 0.0 < lower < upper:
            raise ValueError('Require 0 < lower < upper.')
        lo = float(self.cdf_log10(np.log10(lower)))
        hi = float(self.cdf_log10(np.log10(upper)))
        return max(0.0, min(1.0, hi - lo))

    def logpdf_linear(self, x: np.ndarray | float) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        out = np.full(values.shape, -np.inf, dtype=float)
        ok = np.isfinite(values) & (values > 0.0)
        if np.any(ok):
            logx = np.log10(values[ok])
            out[ok] = self.logpdf_log10(logx) - np.log(values[ok] * np.log(10.0))
        return out

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if n <= 0:
            raise ValueError('n must be positive.')
        p_left = self.sigma_left / (self.sigma_left + self.sigma_right)
        left = rng.random(n) < p_left
        z = np.empty(n, dtype=float)
        n_left = int(np.sum(left))
        if n_left:
            z[left] = self.mu - np.abs(rng.normal(0.0, self.sigma_left, n_left))
        if n_left < n:
            z[~left] = self.mu + np.abs(rng.normal(0.0, self.sigma_right, n - n_left))
        return 10.0 ** z

    def as_record(self) -> dict[str, float | str]:
        return {'label': self.label, 'source': self.source, 'center': self.center, 'lower_68': self.lower_68, 'upper_68': self.upper_68, 'sigma_left_log10': self.sigma_left, 'sigma_right_log10': self.sigma_right, 'interpretation': 'external_posterior_marginal_approximation'}
BRYSON_ETA_HZ_SCENARIOS: Mapping[str, AsymmetricLog10Density] = {'conservative_low': AsymmetricLog10Density(center=0.37, lower_68=0.37 - 0.21, upper_68=0.37 + 0.48, label='Bryson conservative HZ, lower completeness extrapolation', source='Bryson et al. (2021), eta_Earth conservative HZ'), 'conservative_high': AsymmetricLog10Density(center=0.6, lower_68=0.6 - 0.36, upper_68=0.6 + 0.9, label='Bryson conservative HZ, upper completeness extrapolation', source='Bryson et al. (2021), eta_Earth conservative HZ'), 'optimistic_low': AsymmetricLog10Density(center=0.58, lower_68=0.58 - 0.33, upper_68=0.58 + 0.73, label='Bryson optimistic HZ, lower completeness extrapolation', source='Bryson et al. (2021), eta_Earth optimistic HZ'), 'optimistic_high': AsymmetricLog10Density(center=0.88, lower_68=0.88 - 0.51, upper_68=0.88 + 1.28, label='Bryson optimistic HZ, upper completeness extrapolation', source='Bryson et al. (2021), eta_Earth optimistic HZ')}

def get_eta_hz_constraint(name: str) -> AsymmetricLog10Density:
    try:
        return BRYSON_ETA_HZ_SCENARIOS[name]
    except KeyError as exc:
        allowed = ', '.join(BRYSON_ETA_HZ_SCENARIOS)
        raise KeyError(f'Unknown eta_HZ scenario {name!r}. Allowed: {allowed}') from exc
