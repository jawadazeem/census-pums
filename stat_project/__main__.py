from __future__ import annotations

import argparse
from pathlib import Path

from .data import DEFAULT_INPUT, load_workers, weighted_mean, weighted_median
from .modeling import compare_outlier_sensitivity, flag_outliers, run_pca_svd, simple_wage_regression
from .plots import save_pca_variance_plot, save_wage_qq_plot


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Washington workforce backend analysis.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--plots-dir", type=Path, default=Path("figures"))
    parser.add_argument("--skip-modeling", action="store_true")
    args = parser.parse_args()

    print("Loading and cleaning the Washington ACS person file...")
    workers = load_workers(args.input)
    print(f"Cleaned workers: {len(workers):,}")
    print(f"Weighted mean wage: USD {weighted_mean(workers['WAGP'], workers['PWGTP']):,.2f}")
    print(f"Weighted median wage: USD {weighted_median(workers['WAGP'], workers['PWGTP']):,.2f}")

    outliers = flag_outliers(workers)
    print("\nOutlier rules")
    print(outliers["rules"].to_string(index=False))
    save_wage_qq_plot(workers, args.plots_dir / "wage_qq.png")

    if args.skip_modeling:
        print(f"\nSaved Q-Q plot to {args.plots_dir / 'wage_qq.png'}")
        return

    print("\nRunning simple linear regression: wage versus weekly hours...")
    regression = simple_wage_regression(workers)
    print(regression["coefficients"].to_string(index=False))
    print(regression["metrics"].to_string(index=False))
    print("\nWage-outlier sensitivity")
    print(compare_outlier_sensitivity(workers).to_string(index=False))
    print("\nRunning PCA/SVD...")
    pca = run_pca_svd(workers)
    print(pca["components"].to_string(index=False))
    print(pca["summary"].to_string(index=False))
    save_pca_variance_plot(pca, args.plots_dir / "pca_variance.png")
    print(f"\nSaved diagnostic plots under {args.plots_dir}")


if __name__ == "__main__":
    main()
