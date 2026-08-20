from __future__ import annotations

import numpy as np

T_DET_DEFAULT = 200.0   # yr, modern remote technosignature window
T_VIS_DEFAULT = 2000.0  # yr, conservative historical non-visitation window
MODEL3A_TIME_STEPS = 512
MODEL3A_CHUNK_SIZE = 5000


def drake_factor(p: dict[str, np.ndarray]) -> np.ndarray:
    planet_factor = p["eta_hz"] if "eta_hz" in p else p["fp"] * p["ne"]
    return p["R_star"] * planet_factor * p["fl"] * p["fi"] * p["fa"]


def _base_output(model_id, L, N_active, weight=None):
    n = len(L)
    if weight is None:
        weight = np.ones(n)
    return {
        "model_id": np.array(model_id),
        "L_active": L,
        "L_surv": L,
        "N_active": N_active,
        "N_site": N_active,
        "N_source": N_active,
        "N_civ": N_active,
        "Q_site": np.ones(n),
        "kappa_surv": np.zeros(n),
        "weight": weight,
    }


def _soft_cap(x, cap):
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    cap = np.asarray(cap, dtype=float)
    if cap.shape == ():
        cap = np.full_like(x, float(cap))
    cap = np.maximum(cap, 1.0e-300)
    ratio = np.clip(x / cap, 0.0, 700.0)
    return cap * (-np.expm1(-ratio))


def model1(p):
    C_D = drake_factor(p)
    N_active = p["N_active"]
    L = N_active / C_D
    out = _base_output("model1", L, N_active)
    out["C_D"] = C_D
    return out


def model2(p):
    F_astro = p["F_astro"]
    F_bioactive = p["F_bioactive"]
    N_active = p["N_active"]
    L = N_active / (F_astro * F_bioactive)
    out = _base_output("model2", L, N_active)
    out["F_astro"] = F_astro
    out["F_bioactive"] = F_bioactive
    return out


def _exponential_volume_integral(R, beta):
    R = np.asarray(R, dtype=float)
    beta = np.asarray(beta, dtype=float)
    I = np.zeros(np.broadcast_shapes(R.shape, beta.shape), dtype=float)

    Rb = np.broadcast_to(R, I.shape)
    betab = np.broadcast_to(beta, I.shape)
    positive = (Rb > 0.0) & (betab > 0.0)
    if not np.any(positive):
        return I

    x = np.empty_like(I)
    x[positive] = Rb[positive] / betab[positive]


    small = positive & (x < 1.0e-4)
    if np.any(small):
        xs = x[small]
        I[small] = Rb[small] ** 3 * (1.0 / 3.0 - xs / 4.0 + xs * xs / 10.0)


    large = positive & (x > 80.0)
    if np.any(large):
        I[large] = 2.0 * betab[large] ** 3

    mid = positive & ~(small | large)
    if np.any(mid):
        xm = x[mid]
        bm = betab[mid]
        bracket = 2.0 - np.exp(-xm) * (xm * xm + 2.0 * xm + 2.0)
        I[mid] = bm ** 3 * bracket

    return np.maximum(I, 0.0)


def _cumulative_trapezoid(y, x, axis=1):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    if axis != 1:
        raise NotImplementedError("This helper currently supports axis=1 only.")

    dx = np.diff(x, axis=1)
    area = 0.5 * (y[:, 1:] + y[:, :-1]) * dx
    out = np.zeros_like(y)
    out[:, 1:] = np.cumsum(area, axis=1)
    return out


def _forward_model3a_chunk(p_chunk, time_steps=MODEL3A_TIME_STEPS):
    C_D = drake_factor(p_chunk)
    n = len(C_D)

    u = np.linspace(0.0, 1.0, int(time_steps), dtype=float)
    t = p_chunk["T_max"][:, None] * u[None, :]

    tau = np.maximum(t - p_chunk["t_delay"][:, None], 0.0)
    beta = p_chunk["v_eff"][:, None] * tau
    R = p_chunk["R"][:, None]

    rho_set = (
        p_chunk["B_density"]
        * p_chunk["fg"]
        * p_chunk["ne_br"]
        * p_chunk["p_set"]
    )

    I_local = _exponential_volume_integral(R, beta)
    additional_sites = 4.0 * np.pi * rho_set[:, None] * I_local
    N_site_1 = 1.0 + additional_sites

    N50 = np.maximum(p_chunk["N50"][:, None], 1.0e-300)
    gamma = p_chunk["gamma"][:, None]
    redundancy = (np.maximum(additional_sites, 0.0) / N50) ** gamma
    lambda_t = (
        p_chunk["lambda_corr"][:, None]
        + p_chunk["lambda_loc"][:, None] / (1.0 + redundancy)
    )

    H = _cumulative_trapezoid(lambda_t, t, axis=1)
    S = np.exp(-np.clip(H, 0.0, 700.0))

    L_active = np.trapz(S, t, axis=1)
    site_time = np.trapz(S * N_site_1, t, axis=1)
    N_site_bar = site_time / np.maximum(L_active, 1.0e-300)

    N_source_raw = C_D * L_active
    N_active_raw = C_D * site_time

    N_gal_cap = p_chunk.get("N_gal_cap", np.full_like(N_active_raw, np.inf))
    if np.all(np.isinf(N_gal_cap)):
        N_active = N_active_raw
        N_source = N_source_raw
    else:
        N_active = _soft_cap(N_active_raw, N_gal_cap)
        N_source = _soft_cap(N_source_raw, N_gal_cap)
    cap_fraction = N_active / np.maximum(N_gal_cap, 1.0e-300)


    lambda_initial = p_chunk["lambda_corr"] + p_chunk["lambda_loc"]
    lambda_final = lambda_t[:, -1]
    N_site_final = N_site_1[:, -1]
    beta_final = beta[:, -1]

    return {
        "C_D": C_D,
        "L_active": L_active,
        "L_surv": L_active,
        "N_active": N_active,
        "N_site": N_active,
        "N_active_raw": N_active_raw,
        "N_site_raw": N_active_raw,
        "N_source_raw": N_source_raw,
        "N_gal_cap": N_gal_cap,
        "cap_fraction": cap_fraction,
        "N_source": N_source,
        "N_civ": N_source,
        "Q_site": N_site_bar,
        "N_site_bar": N_site_bar,
        "site_time": site_time,
        "rho_set": rho_set,
        "lambda_initial": lambda_initial,
        "lambda_final": lambda_final,
        "N_site_final": N_site_final,
        "beta_final": beta_final,
        "weight": np.ones(n),
    }


def model3a(p):
    n = len(p["R_star"])
    pieces = []

    for start in range(0, n, MODEL3A_CHUNK_SIZE):
        stop = min(start + MODEL3A_CHUNK_SIZE, n)
        p_chunk = {key: np.asarray(value)[start:stop] for key, value in p.items()}
        pieces.append(_forward_model3a_chunk(p_chunk))

    out = {"model_id": np.array("model3a")}
    keys = pieces[0].keys()
    for key in keys:
        out[key] = np.concatenate([piece[key] for piece in pieces])

    out["kappa_surv"] = np.zeros(n)
    return out


def _constant_hazard_lifetime(lambda_eff, T_max):
    lambda_eff = np.asarray(lambda_eff, dtype=float)
    T_max = np.asarray(T_max, dtype=float)
    x = np.clip(lambda_eff * T_max, 0.0, 700.0)

    finite = lambda_eff > 0.0
    L = np.empty_like(lambda_eff)
    L[finite] = -np.expm1(-x[finite]) / lambda_eff[finite]
    L[~finite] = T_max[~finite]
    return L


def model3b(p):
    C_D = drake_factor(p)
    M_site = np.maximum(p["M_site"], 1.0)

    additional_sites = np.maximum(M_site - 1.0, 0.0)
    redundancy = (additional_sites / np.maximum(p["N50"], 1.0e-300)) ** p["gamma"]

    lambda_base = p["lambda_corr"] + p["lambda_loc"] / (1.0 + redundancy)

    T_det = p.get("T_det", np.full_like(lambda_base, T_DET_DEFAULT))
    T_vis = p.get("T_vis", np.full_like(lambda_base, T_VIS_DEFAULT))
    chi_vis = p.get("chi_vis", np.ones_like(lambda_base))
    T_silence = T_det + chi_vis * T_vis

    T50_silence = np.maximum(p.get("T50_silence", np.full_like(lambda_base, 200.0)), 1.0e-300)
    eta_silence = np.maximum(p.get("eta_silence", np.ones_like(lambda_base)), 0.0)
    silence_gain = np.power(1.0 + T_silence / T50_silence, eta_silence)
    silence_gain = np.maximum(silence_gain, 1.0)

    lambda_haz = lambda_base / silence_gain

    T_max = p["T_max"]
    L_active = _constant_hazard_lifetime(lambda_haz, T_max)

    N_source_raw = C_D * L_active
    N_site_raw = N_source_raw * M_site
    N_gal_cap = p.get("N_gal_cap", np.full_like(N_site_raw, np.inf))
    if np.all(np.isinf(N_gal_cap)):
        N_site = N_site_raw
        N_source = N_source_raw
    else:
        N_site = _soft_cap(N_site_raw, N_gal_cap)
        N_source = _soft_cap(N_source_raw, N_gal_cap)
    Q_site = M_site
    cap_fraction = N_site / np.maximum(N_gal_cap, 1.0e-300)

    lambda_vis = p.get("lambda_vis", np.zeros_like(L_active))
    E_obs = p["lambda_det"] * T_det + lambda_vis * T_vis
    mu_obs = N_site * E_obs
    P_zero = np.exp(-np.clip(mu_obs, 0.0, 700.0))

    out = _base_output("model3b", L_active, N_site, weight=np.ones_like(L_active))
    out["L_surv"] = L_active
    out["N_source"] = N_source
    out["N_civ"] = N_source
    out["N_site"] = N_site
    out["N_active_raw"] = N_site_raw
    out["N_site_raw"] = N_site_raw
    out["N_source_raw"] = N_source_raw
    out["N_gal_cap"] = N_gal_cap
    out["cap_fraction"] = cap_fraction
    out["Q_site"] = Q_site
    out["kappa_surv"] = np.zeros_like(L_active)
    out["C_D"] = C_D
    out["M_site"] = M_site
    out["additional_sites"] = additional_sites
    out["redundancy"] = redundancy
    out["lambda_loc"] = p["lambda_loc"]
    out["lambda_corr"] = p["lambda_corr"]
    out["lambda_base"] = lambda_base
    out["lambda_haz"] = lambda_haz
    out["lambda_eff"] = lambda_haz  
    out["N50"] = p["N50"]
    out["gamma"] = p["gamma"]
    out["T_max"] = T_max
    out["T_det"] = T_det
    out["T_vis"] = T_vis
    out["chi_vis"] = chi_vis
    out["T_silence"] = T_silence
    out["T50_silence"] = T50_silence
    out["eta_silence"] = eta_silence
    out["silence_gain"] = silence_gain
    out["lambda_det"] = p["lambda_det"]
    out["lambda_vis"] = lambda_vis
    out["E_obs"] = E_obs
    out["mu_obs"] = mu_obs
    out["mu_det"] = mu_obs  
    out["P_zero"] = P_zero
    return out



def model4(p):
    numerator = p["N_star_ne"] * p["fg"] * p["fpm"] * p["fm"] * p["fj"] * p["fme"]
    denominator = p["R_star"] * p["ne_den"]
    L = numerator / denominator
    N_active = p["N_active"]
    out = _base_output("model4", L, N_active)
    out["N_star_ne"] = p["N_star_ne"]
    return out


MODELS = {
    "model1": model1,
    "model2": model2,
    "model3a": model3a,
    "model3b": model3b,
    "model4": model4,
}
