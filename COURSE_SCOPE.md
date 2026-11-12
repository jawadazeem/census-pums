# Course-aligned backend scope

The modeling code follows the supplied CS/STAT 3654 lectures.

| Backend work | Lecture basis |
|---|---|
| Pearson and Spearman correlation | Statistics and correlation lectures |
| Q-Q and normality checks | Statistics and project requirements |
| IQR outlier flags | Statistics and correlation assumptions |
| OLS, train/test split, MSE, R-squared, overall F-test | Lecture 11 |
| Ridge, lasso, elastic net, five-fold cross-validation | Lecture 11 |
| VIF and condition number | Linear algebra refresher and Lecture 11 |
| Standardization, PCA, SVD, explained variance, loadings | Lecture 10 and linear algebra refresher |

Classification and clustering are intentionally outside this branch.

Survey weights are used for population descriptions. The regression and PCA steps analyze the PUMS sample rows, matching the modeling examples in the lectures.

For modeling, explicit work-from-home records receive zero commute minutes. Only records with genuinely unknown commute mode require median imputation.
