import itertools
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_all_pairs_heatmaps(df, param_list, metric_column="best_val_loss", bins=5):
    """
    Plots a grid of heatmaps for each pair of hyperparameters.
    Each heatmap shows how the mean of 'metric_column' changes
    across the binned values of the two hyperparameters.

    :param df: pandas DataFrame, e.g., loaded from "optuna_results.csv"
    :param param_list: List of hyperparameter column names to consider
    :param metric_column: The column in 'df' representing your performance metric
    :param bins: Number of bins for discretizing continuous hyperparams
    """
    # Generate all pairs (2-combinations) of parameters
    pairs = list(itertools.combinations(param_list, 2))
    n_pairs = len(pairs)

    # Decide on subplot layout (e.g., 2 or 3 columns wide)
    # For a neat layout, we can do ~3 columns, or adapt as needed
    cols = min(n_pairs, 3)  # e.g., up to 3 columns
    rows = (n_pairs + cols - 1) // cols  # ceiling division

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    # If 'df' is large or has mostly continuous hyperparams,
    # consider whether to do binning or not.
    # For each pair, we'll create a pivot table and plot a heatmap.
    for idx, (param_x, param_y) in enumerate(pairs):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]

        # If either param is continuous, we can bin it
        # (You can skip binning if your parameters are purely discrete.)
        df_plot = df.copy()
        # Bin both hyperparameters (this will convert them to categorical)
        df_plot[f"{param_x}_bin"] = pd.cut(df_plot[param_x], bins=bins)
        df_plot[f"{param_y}_bin"] = pd.cut(df_plot[param_y], bins=bins)

        # Group by the binned columns and compute mean (or min, max) of the metric
        pivot_df = (
            df_plot
            .groupby([f"{param_x}_bin", f"{param_y}_bin"])[metric_column]
            .mean()
            .reset_index()
        )

        # Pivot to get matrix form for heatmap
        pivot_df = pivot_df.pivot(index=f"{param_x}_bin", columns=f"{param_y}_bin", values=metric_column)

        # Plot the heatmap
        sns.heatmap(pivot_df, annot=True, cmap="viridis", ax=ax)
        ax.set_title(f"{param_x} vs. {param_y} ({metric_column})")
        ax.set_xlabel(param_y)
        ax.set_ylabel(param_x)

    # If we have any unused subplots (in case #pairs < rows*cols), hide them
    for j in range(idx + 1, rows * cols):
        r = j // cols
        c = j % cols
        axes[r][c].axis("off")

    plt.tight_layout()
    plt.show()


# --------------------- USAGE  ---------------------
if __name__ == "__main__":

    csv_path = r"data/results/optuna_results_2025-01-12_00-12-57.csv"
    # Suppose you have a CSV "optuna_results.csv" with columns like:
    # trial_number, learning_rate, batch_size, dropout, dense_layers, best_val_loss, best_val_f1, state
    df = pd.read_csv(csv_path)

    # Define which hyperparameters you'd like to visualize
    param_list = ["learning_rate", "dropout", "batch_size", "dense_layers"]

    # The performance metric you want to display
    metric_column = "best_val_loss"  # or "best_val_f1"

    # Plot a global grid of heatmaps for each pair of hyperparameters
    plot_all_pairs_heatmaps(df, param_list, metric_column=metric_column, bins=20)
