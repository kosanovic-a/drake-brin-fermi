from __future__ import annotations

'Run all four Bryson eta_HZ scenarios with a common baseline seed.\n\nThe weak-prior sensitivity analysis is executed only for the first scenario by\ndefault because it does not depend on the selected external eta_HZ marginal.\n'
import argparse
from pathlib import Path
from observation_informed import run_comment2
from observational_constraints import BRYSON_ETA_HZ_SCENARIOS
from sampler import DISTRIBUTIONS

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run all conservative/optimistic Comment 2 occurrence-rate scenarios.')
    parser.add_argument('--out', default='outputs_comment2_all')
    parser.add_argument('--samples', type=int, default=120000)
    parser.add_argument('--dist', choices=DISTRIBUTIONS, default='lognormal')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--sensitivity-samples', type=int, default=20000)
    parser.add_argument('--skip-seti-sensitivity', action='store_true')
    parser.add_argument('--seti-logq-min', type=float, default=-14.0)
    parser.add_argument('--seti-logq-max', type=float, default=-2.0)
    parser.add_argument('--seti-grid-points', type=int, default=25)
    parser.add_argument('--n-galactic-targets', type=float, default=100000000000.0)
    parser.add_argument('--settlement-bridge', choices=['site', 'source'], default='site')
    parser.add_argument('--repeat-prior-sensitivity', action='store_true', help='Recompute weak-prior sensitivity in every scenario directory.')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    root = Path(args.out)
    scenarios = tuple(BRYSON_ETA_HZ_SCENARIOS)
    for index, scenario in enumerate(scenarios):
        sensitivity_samples = args.sensitivity_samples if index == 0 or args.repeat_prior_sensitivity else 0
        print(f'[INFO] Running Comment 2 scenario: {scenario}')
        run_comment2(outdir=root / scenario, samples=args.samples, dist=args.dist, seed=args.seed, eta_scenario=scenario, sensitivity_samples=sensitivity_samples, run_seti_sensitivity=not args.skip_seti_sensitivity, seti_logq_min=args.seti_logq_min, seti_logq_max=args.seti_logq_max, seti_grid_points=args.seti_grid_points, n_galactic_targets=args.n_galactic_targets, settlement_bridge=args.settlement_bridge, allow_model3b_seti=False)
    print(f'[OK] All scenarios written below {root.resolve()}')
if __name__ == '__main__':
    main()
