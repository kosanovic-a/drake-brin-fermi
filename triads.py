TRIADS = {
    "model1": {
        "R_star": [1.0, 2.0, 5.0],          # yr^-1
        "fp": [0.7, 0.95, 1.0],             # fraction of stars with planets
        "ne": [0.2, 0.6, 1.5],              # HZ rocky planets per planetary system
        "fl": [1.0e-2, 0.5, 1.0],           # fraction where life emerges
        "fi": [1.0e-4, 0.1, 1.0],           # fraction where intelligence emerges
        "fa": [1.0e-3, 0.1, 1.0],           # active technological fraction
        "N_active": [1.0, 3.0, 1.0e3],      # active civilizations
    },

    "model2": {
        "F_astro": [0.14, 1.14, 7.5],       # yr^-1
        "F_bioactive": [1.0e-9, 5.0e-3, 1.0],
        "N_active": [1.0, 3.0, 1.0e3],      # active civilizations
    },

    "model3a": {
        "R_star": [1.0, 2.0, 5.0],          # yr^-1
        "fp": [0.7, 0.95, 1.0],
        "ne": [0.2, 0.6, 1.5],
        "fl": [1.0e-2, 0.5, 1.0],
        "fi": [1.0e-4, 0.1, 1.0],
        "fa": [1.0e-3, 0.1, 1.0],

        "B_density": [1.0e-3, 3.0e-3, 1.0e-2],   # ly^-3
        "fg": [0.05, 0.10, 0.50],                 # suitable Galactic environments
        "ne_br": [0.1, 0.6, 3.0],                 # settleable targets per suitable system
        "p_set": [1.0e-6, 1.0e-3, 1.0e-1],        # reached target becomes durable site

        "v_eff": [1.0e-5, 1.0e-3, 1.0e-1],        # ly yr^-1
        "R": [2.0e4, 3.0e4, 5.0e4],               # ly, maximum spatial horizon
        "t_delay": [1.0e2, 1.0e4, 1.0e6],         # yr before durable settlement begins

        "lambda_loc": [1.0e-6, 1.0e-5, 1.0e-4],   # yr^-1, local single-site hazard
        "lambda_corr": [1.0e-9, 1.0e-7, 1.0e-5],  # yr^-1, correlated/global floor
        "N50": [1.0, 1.0e3, 1.0e6],               # added sites for half-reduction
        "gamma": [0.5, 1.0, 2.0],                 # redundancy strength
        "T_max": [1.0e6, 1.0e7, 1.0e8],           # yr, survival integration cutoff
        "N_gal_cap": [1.0e10, 1.0e11, 1.0e12],    # finite Galactic active-site capacity
    },

    "model3b": {
        "R_star": [1.0, 2.0, 5.0],          # yr^-1
        "fp": [0.7, 0.95, 1.0],
        "ne": [0.2, 0.6, 1.5],
        "fl": [1.0e-2, 0.5, 1.0],
        "fi": [1.0e-4, 0.1, 1.0],
        "fa": [1.0e-3, 0.1, 1.0],

        "M_site": [1.0, 1.0e3, 1.0e6],           # active sites per source civilization
        "N_gal_cap": [1.0e10, 1.0e11, 1.0e12],   # finite Galactic active-site capacity
        "lambda_loc": [1.0e-6, 1.0e-5, 1.0e-4],  # yr^-1
        "lambda_corr": [1.0e-9, 1.0e-7, 1.0e-5], # yr^-1
        "N50": [1.0, 1.0e3, 1.0e6],              # added sites for half-reduction
        "gamma": [0.5, 1.0, 2.0],                # redundancy strength
        "T_max": [1.0e6, 1.0e7, 1.0e8],          # yr

        "T_det": [50.0, 200.0, 500.0],
        "T_vis": [500.0, 2000.0, 5000.0],
        "chi_vis": [0.1, 1.0, 2.0],
        "T50_silence": [50.0, 200.0, 2000.0],
        "eta_silence": [0.5, 1.0, 1.5],

        "lambda_det": [1.0e-13, 1.0e-10, 1.0e-7],
        "lambda_vis": [1.0e-15, 1.0e-12, 1.0e-9],
    },

    "model4": {
        "N_star_ne": [2.0e10, 1.0e11, 6.0e11],
        "fg": [0.05, 0.10, 0.50],
        "fpm": [0.01, 0.10, 0.20],
        "fm": [0.003, 0.01, 0.03],
        "fj": [0.05, 0.20, 0.80],
        "fme": [0.003, 0.01, 0.03],
        "R_star": [1.0, 2.0, 5.0],
        "ne_den": [0.2, 0.6, 1.5],
        "N_active": [1.0, 3.0, 1.0e3],
    },
}

MODEL_NAMES = ["model1", "model2", "model3a", "model3b", "model4"]


if __name__ == "__main__":
    import json
    from pathlib import Path

    out = Path("triads.json")
    out.write_text(json.dumps(TRIADS, indent=4), encoding="utf-8")
    print(f"[OK] Wrote {out.resolve()}")
