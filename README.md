# Drake-Fermi-Brin probabilistic framework

This repository contains the Python pipeline used for the Monte Carlo Drake-Fermi-Brin model comparison and the additional analyses prepared for the revised manuscript.

The code is intentionally script-based and kept in one directory. Each stage writes explicit numerical or graphical outputs below `outputs/`, so the generated samples, tables, figures, and diagnostics can be inspected independently.

## Repository structure

| File | Purpose |
|---|---|
| `main.py` | Main command-line pipeline for model generation, plotting, supermodels, PCA, summaries, importance grids, and distribution figures. |
| `triads.py` | Bounded parameter triads for the five basic models. |
| `sampler.py` | Bounded lognormal, log-linear, Gaussian, and mixed sampling. |
| `models.py` | Implementations of Models I, II, IIIA, IIIB-M, and IV. |
| `supermodels.py` | SM1-SM5 construction and supermodel summaries. |
| `plots.py` | Heatmaps, surfaces, marginal densities, and survival plots. |
| `pca_analysis.py` | PCA and exploratory KMeans diagnostics. |
| `importance_grid.py` | Within-model Random-Forest permutation importance. |
| `result_summary.py` | Quantile summaries and interval figures. |
| `distributions.py` | Sampling-distribution comparison figure. |
| `figure_style.py` | Matplotlib and LaTeX figure settings. |
| `regime_masses.py` | Fixed diagnostic-region mass tables. |
| `table3_robustness.py` | Bound, peak, and distribution-family robustness analysis. |
| `triad_sensitivity.py` | Utilities for triad perturbation scenarios. |
| `observation_informed.py` | Exoplanet-informed propagation and SETI reweighting. |
| `run_observation_informed.py` | Runs the four observation-informed scenarios. |
| `observational_constraints.py` | External occurrence-rate constraints. |
| `bayesian_reweighting.py` | Importance weighting and effective-sample-size calculations. |
| `seti_observation.py` | Reduced SETI null-detection observation layer. |
| `semantic_bridge_bma.py` | Site/source semantic bridges and Bayesian model averaging. |
| `layered_variance.py` | Layered variance decomposition and stacked-bar figure. |
| `cluster_validation.py` | Elbow, silhouette, stability, gap, and Gaussian-mixture diagnostics. |
| `clustering_sensitivity.py` | Combined clustering and diagnostic-threshold analysis. |
| `regime_sensitivity.py` | Sensitivity of fixed diagnostic-region thresholds. |
| `configuration_counts.py` | Counts model and supermodel distributional configurations. |
| `analysis_utils.py` | Shared revision-analysis utilities. |
| `run_revision.py` | Combined runner for revision-specific analyses. |

## Requirements

The repository was checked with Python 3.13.5 and the package versions listed in `requirements.txt`.

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

Linux or macOS activation:

```bash
source .venv/bin/activate
```

A LaTeX installation such as MiKTeX or TeX Live is optional. Set `CIV_USE_TEX=1` to use LaTeX text rendering or `CIV_USE_TEX=0` to force Matplotlib mathtext.

## Baseline workflow

Run the complete baseline workflow:

```bash
python main.py clean generate plot super pca summary importance distributions
python regime_masses.py --out outputs
```

Generated `.npz` files are preserved unless `--delete-npz` is supplied. Keep them when running PCA, summaries, semantic-bridge analysis, layered variance decomposition, or reproducibility checks.

The default basic-model distribution is `lognormal`. The historical supermodel default is `mixed`, and all SM1-SM5 scenarios use the same component bank within a selected supermodel run.

## Revision analyses

### Configuration counts

```bash
python configuration_counts.py
```

### Bounded-triad robustness

```bash
python table3_robustness.py --samples 100000
```

The script uses five seeds and common random numbers within each seed and writes seed-level and averaged summaries.

### Observation-informed analysis

```bash
python run_observation_informed.py --out outputs/revision/observation --samples 120000 --seed 12345
```

### Semantic bridges and Bayesian model averaging

```bash
python main.py clean generate
python semantic_bridge_bma.py --sample-dir outputs/samples --out-dir outputs/revision/semantic_bma
```

### Layered uncertainty decomposition

```bash
python layered_variance.py --sample-dir outputs/samples --out outputs/revision/layered_variance
```

This produces the variance-component summaries and the normalized stacked-bar figure used for the layered-uncertainty analysis.

### Clustering and threshold sensitivity

```bash
python main.py clean generate super
python clustering_sensitivity.py --out outputs
```

### Combined revision runner

```bash
python run_revision.py counts table3 semantic variance --out outputs
```

Observation-informed and clustering actions can be added when those complete analyses are required.

## Reproducibility map

| Manuscript output | Script | Main output |
|---|---|---|
| Basic-model figures | `main.py generate plot` | `outputs/samples`, `outputs/figures` |
| Supermodel figures | `main.py super` | `outputs/super_samples`, `outputs/super_figures` |
| Diagnostic-region masses | `regime_masses.py` | CSV and LaTeX table |
| Triad robustness | `table3_robustness.py` | Seed and summary CSV files |
| Observation-informed results | `run_observation_informed.py` | Scenario summaries and stored ensembles |
| Semantic bridges and BMA | `semantic_bridge_bma.py` | CSV, JSON, and LaTeX summaries |
| Layered uncertainty figure | `layered_variance.py` | PDF, CSV, JSON, and LaTeX summaries |
| Clustering diagnostics | `clustering_sensitivity.py` | Validation tables and figures |
| Configuration counts | `configuration_counts.py` | Configuration-count summary |

The numerical pipeline is deterministic for a fixed seed, sample count, distribution choice, and code version. For an archived release, retain the final code, seeds, generated CSV files, relevant `.npz` ensembles, and the manuscript version based on those outputs.

## Implementation notes

`models.py` accepts the baseline factorized planetary term `fp * ne` and the observation-informed aggregate `eta_hz`. Baseline behavior is unchanged.

For forward settlement Models IIIA and IIIB-M, output-derived abundance is excluded from the Random-Forest predictor set. For inverse Models I and II, the sampled abundance coordinate remains an input.

The layered variance analysis uses equal model and semantic-bridge representation. Its variance shares describe the declared computational ensemble and are not posterior probabilities of the physical models.

The SETI update is conditional on the selected survey assumptions and effective-detectability prior. It does not identify individual biological or technological parameters from a null result.

## Validation

The merged code, clustering and other workflows were checked.

## Citation and license

Citation metadata are provided in `CITATION.cff`. 

The software is distributed under the MIT License in `LICENSE.md`.
