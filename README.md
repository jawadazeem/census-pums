# Washington ACS backend

This backend loads the Washington ACS person file, cleans the worker population, and exposes analysis functions for Dash callbacks.

## Run

    python -m pip install -r requirements.txt
    python -m stat_project --input C:\path\to\psam_p53.csv

Use --skip-modeling for a quick cleaning and outlier demonstration.

The full command prints regression, the overall F-test, the wage-outlier sensitivity comparison, VIF, and PCA/SVD results. It also saves:

- figures/wage_qq.png
- figures/pca_variance.png

The frontend imports functions from stat_project.services directly. CSV files are not used as the application interface.
