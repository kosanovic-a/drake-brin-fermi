from __future__ import annotations

'Transparent target-level SETI null-detection sensitivity model.\n\nThis module is intentionally narrower than the general haystack likelihood in\nAppendix F of the manuscript.  It converts a global simulated population into\nan expected count for a target list under an explicit occupancy approximation.\nThe unknown transmitter and survey-selection factors are collapsed into\n``q_eff`` and must be varied or marginalized; they are not estimated by the\nnull result itself.\n'
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp

@dataclass(frozen=True)
class TargetSurvey:
    name: str
    n_targets: int
    n_detections: int
    frequency_min_mhz: float
    frequency_max_mhz: float
    integration_hours: float
    channel_width_hz: float
    search_fraction_context: float | None
    source: str
    target_count_is_rounded: bool = False

    def __post_init__(self) -> None:
        if self.n_targets <= 0:
            raise ValueError('n_targets must be positive.')
        if self.n_detections < 0:
            raise ValueError('n_detections must be nonnegative.')
        if self.frequency_min_mhz >= self.frequency_max_mhz:
            raise ValueError('Invalid survey frequency interval.')
MWA_VELA_2020 = TargetSurvey(name='MWA Vela 2020', n_targets=10000000, n_detections=0, frequency_min_mhz=98.0, frequency_max_mhz=128.0, integration_hours=17.0, channel_width_hz=10000.0, search_fraction_context=2e-16, source='Tremblay and Tingay (2020)', target_count_is_rounded=True)

def target_occupancy_probability(population: np.ndarray, *, q_eff: float | np.ndarray, n_galactic_targets: float) -> np.ndarray:
    pop = np.asarray(population, dtype=float)
    q = np.asarray(q_eff, dtype=float)
    if np.any(~np.isfinite(pop)) or np.any(pop < 0.0):
        raise ValueError('population must be finite and nonnegative.')
    if np.any(~np.isfinite(q)) or np.any((q < 0.0) | (q > 1.0)):
        raise ValueError('q_eff must lie in [0, 1].')
    if not np.isfinite(n_galactic_targets) or n_galactic_targets <= 0.0:
        raise ValueError('n_galactic_targets must be finite and positive.')
    mean_per_target = pop * q / float(n_galactic_targets)
    return -np.expm1(-np.clip(mean_per_target, 0.0, 700.0))

def expected_detections(population: np.ndarray, survey: TargetSurvey, *, q_eff: float | np.ndarray, n_galactic_targets: float=100000000000.0, recovery_efficiency: float=1.0) -> np.ndarray:
    if not 0.0 <= recovery_efficiency <= 1.0:
        raise ValueError('recovery_efficiency must lie in [0, 1].')
    p_host = target_occupancy_probability(population, q_eff=q_eff, n_galactic_targets=n_galactic_targets)
    return survey.n_targets * recovery_efficiency * p_host

def poisson_log_likelihood(count: int, rate: np.ndarray) -> np.ndarray:
    from scipy.special import gammaln
    lam = np.asarray(rate, dtype=float)
    if count < 0:
        raise ValueError('count must be nonnegative.')
    if np.any(~np.isfinite(lam)) or np.any(lam < 0.0):
        raise ValueError('rate must be finite and nonnegative.')
    if count == 0:
        return -lam
    with np.errstate(divide='ignore'):
        return count * np.log(lam) - lam - gammaln(count + 1.0)

def survey_log_likelihood(population: np.ndarray, survey: TargetSurvey=MWA_VELA_2020, *, q_eff: float, n_galactic_targets: float=100000000000.0, recovery_efficiency: float=1.0) -> np.ndarray:
    lam = expected_detections(population, survey, q_eff=q_eff, n_galactic_targets=n_galactic_targets, recovery_efficiency=recovery_efficiency)
    return poisson_log_likelihood(survey.n_detections, lam)

def marginalized_log_likelihood_loguniform_q(population: np.ndarray, survey: TargetSurvey=MWA_VELA_2020, *, log10_q_min: float, log10_q_max: float, n_galactic_targets: float=100000000000.0, recovery_efficiency: float=1.0, quadrature_nodes: int=48, chunk_size: int=20000) -> np.ndarray:
    if not log10_q_min < log10_q_max <= 0.0:
        raise ValueError('Require log10_q_min < log10_q_max <= 0.')
    if quadrature_nodes < 4:
        raise ValueError('quadrature_nodes must be at least 4.')
    pop = np.asarray(population, dtype=float).reshape(-1)
    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_nodes))
    mid = 0.5 * (log10_q_min + log10_q_max)
    half = 0.5 * (log10_q_max - log10_q_min)
    log10_q = mid + half * nodes
    q = 10.0 ** log10_q
    log_quad_weights = np.log(0.5 * weights)
    result = np.empty(pop.size, dtype=float)
    for start in range(0, pop.size, int(chunk_size)):
        stop = min(start + int(chunk_size), pop.size)
        p = pop[start:stop, None]
        lam = expected_detections(p, survey, q_eff=q[None, :], n_galactic_targets=n_galactic_targets, recovery_efficiency=recovery_efficiency)
        log_like = poisson_log_likelihood(survey.n_detections, lam)
        result[start:stop] = logsumexp(log_like + log_quad_weights[None, :], axis=1)
    return result

def semantic_population_from_output(output: dict[str, np.ndarray], model_id: str, *, settlement_bridge: str='site') -> np.ndarray:
    if model_id in {'model1', 'model2', 'model4'}:
        key = 'N_civ' if 'N_civ' in output else 'N_active'
    elif model_id in {'model3a', 'model3b'}:
        if settlement_bridge == 'site':
            key = 'N_site'
        elif settlement_bridge == 'source':
            key = 'N_source'
        else:
            raise ValueError("settlement_bridge must be 'site' or 'source'.")
    else:
        raise KeyError(f'Unknown model_id {model_id!r}.')
    if key not in output:
        raise KeyError(f'Output for {model_id} does not contain semantic field {key!r}.')
    return np.asarray(output[key], dtype=float)
