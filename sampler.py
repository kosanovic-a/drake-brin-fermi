from __future__ import annotations

import numpy as np


BASE_DISTRIBUTIONS = ("lognormal", "loglinear", "gauss")
DISTRIBUTIONS = BASE_DISTRIBUTIONS + ("mixed",)
DIST_TO_CODE = {name: i for i, name in enumerate(BASE_DISTRIBUTIONS)}
CODE_TO_DIST = np.array(BASE_DISTRIBUTIONS, dtype="U16")


def _check_triad(triad: list[float] | tuple[float, float, float]) -> tuple[float, float, float]:
    lo, peak, hi = map(float, triad)
    if not (lo > 0.0 and peak > 0.0 and hi > 0.0):
        raise ValueError(f"Triad values must be positive, got {triad}.")
    if not (lo <= peak <= hi):
        raise ValueError(f"Triad must satisfy min <= peak <= max, got {triad}.")
    if lo == hi:
        raise ValueError(f"Degenerate triad, got {triad}.")
    return lo, peak, hi


def _resample_until_inside(values, draw, lo, hi, max_iter=100):
    mask = (values < lo) | (values > hi)
    it = 0
    while np.any(mask) and it < max_iter:
        values[mask] = draw(int(np.sum(mask)))
        mask = (values < lo) | (values > hi)
        it += 1
    return np.clip(values, lo, hi)


def sample_triad(triad, n: int, dist: str, rng: np.random.Generator) -> np.ndarray:
    lo, peak, hi = _check_triad(triad)

    if dist == "loglinear":
        a, c, b = np.log10([lo, peak, hi])
        return 10.0 ** rng.triangular(a, c, b, size=n)

    if dist == "lognormal":
        a, c, b = np.log([lo, peak, hi])
        sigma = max(abs(c - a), abs(b - c)) / 3.0
        if sigma <= 0.0:
            raise ValueError(f"Invalid lognormal sigma for triad {triad}.")

        def draw(k):
            return rng.normal(c, sigma, size=k)

        y = draw(n)
        y = _resample_until_inside(y, draw, a, b)
        return np.exp(y)

    if dist == "gauss":
        sigma = max(abs(peak - lo), abs(hi - peak)) / 3.0
        if sigma <= 0.0:
            raise ValueError(f"Invalid Gaussian sigma for triad {triad}.")

        def draw(k):
            return rng.normal(peak, sigma, size=k)

        return _resample_until_inside(draw(n), draw, lo, hi)

    raise ValueError(f"Unknown distribution {dist!r}. Allowed: {DISTRIBUTIONS}.")


def sample_triad_mixed(triad, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    codes = rng.integers(0, len(BASE_DISTRIBUTIONS), size=n, dtype=np.int8)
    values = np.empty(n, dtype=float)

    for code, dist in enumerate(BASE_DISTRIBUTIONS):
        mask = codes == code
        count = int(np.sum(mask))
        if count > 0:
            values[mask] = sample_triad(triad, count, dist, rng)

    return values, codes


def sample_params(
    triads: dict,
    n: int,
    dist: str,
    rng: np.random.Generator,
    return_dist_codes: bool = False,
):

    if dist not in DISTRIBUTIONS:
        raise ValueError(f"Unknown distribution {dist!r}. Allowed: {DISTRIBUTIONS}.")

    params = {}
    dist_codes = {}

    for name, triad in triads.items():
        if dist == "mixed":
            values, codes = sample_triad_mixed(triad, n, rng)
        else:
            values = sample_triad(triad, n, dist, rng)
            codes = np.full(n, DIST_TO_CODE[dist], dtype=np.int8)

        params[name] = values
        dist_codes[f"dist_{name}"] = codes

    if return_dist_codes:
        return params, dist_codes
    return params
