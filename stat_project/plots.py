from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def save_wage_qq_plot(data, output_path: str | Path, random_state: int = 3654) -> Path:
    wages = data["WAGP"].dropna().sample(min(5_000, data["WAGP"].notna().sum()), random_state=random_state)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    stats.probplot(wages, dist="norm", plot=axes[0])
    axes[0].set_title("Annual wage")
    stats.probplot(np.log1p(wages), dist="norm", plot=axes[1])
    axes[1].set_title("Log-transformed annual wage")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.suptitle("Q-Q plots for annual wage")
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def save_pca_variance_plot(pca_results: dict, output_path: str | Path) -> Path:
    table = pca_results["components"]
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.bar(table["component"], table["explained_variance_ratio"], label="Explained variance")
    axis.plot(
        table["component"], table["cumulative_explained_variance"],
        marker="o", color="tab:red", label="Cumulative variance",
    )
    axis.set_xlabel("Principal component")
    axis.set_ylabel("Variance ratio")
    axis.set_ylim(0, 1.05)
    axis.set_title("PCA explained variance")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path
