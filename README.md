# Census Washington Workforce

This backend cleans the Washington ACS person file and provides functions the Dash callbacks can call directly.

## setup

Use your existing Conda environment:

```bash
conda activate YOUR_ENVIRONMENT
conda install -c conda-forge pandas numpy scipy scikit-learn
python scripts/extract_data.py
```

## run the demo

```bash
python -m backend
```

The demo prints cleaned rows, summary statistics, outlier rules, regression results, VIF, and PCA/SVD results. Advanced calculations are marked as work in progress, with placeholders for conclusions that still need review.

## use from Dash

```python
from backend.services import load_dashboard_data, filter_workers
from backend.services import regression_analysis, pca_svd_analysis

workers = load_dashboard_data()
filtered = filter_workers(workers, age_range=[25, 55])
regression = regression_analysis(filtered)
pca = pca_svd_analysis(filtered)
```

`backend.statistics` also contains the lower-level analysis functions.
