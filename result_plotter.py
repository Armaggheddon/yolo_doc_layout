from pathlib import Path
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

PLOTS_ROOT = Path(__file__).parent / "plots"
PLOTS_ROOT.mkdir(exist_ok=True)

LABELS = ["all", "caption", "footnote", "formula", "list-item", "page-footer", "page-header", "picture", "section-header", "table", "text", "title"]
LABEL2ID = {k: v for v, k in enumerate(LABELS)}

@dataclass
class Metrics:
    model_size: str
    box_precision: list[float]
    recall: list[float]
    map50: list[float]
    map50_95: list[float]

    def __getitem__(self, key):
        """ Return the metrics for a given label key."""
        if key in LABEL2ID:
            idx = LABEL2ID[key]
            return [
                self.box_precision[idx],
                self.recall[idx],
                self.map50[idx],
                self.map50_95[idx],
            ]
    
    def get_metric(self, metric_name: str):
        """ Return the metric values for all labels."""
        if metric_name == "box_precision":
            return self.box_precision
        elif metric_name == "recall":
            return self.recall
        elif metric_name == "map50":
            return self.map50
        elif metric_name == "map50_95":
            return self.map50_95
        else:
            raise ValueError(f"Unknown metric name: {metric_name}")
        
    @staticmethod
    def available_metrics():
        return ["box_precision", "recall", "map50", "map50_95"]
    
yolo26n = Metrics(
    "yolo26n",
    [0.901, 0.943, 0.905, 0.89, 0.921, 0.902, 0.93, 0.875, 0.924, 0.858, 0.935, 0.825],
    [0.848, 0.869, 0.718, 0.825, 0.907, 0.92, 0.84, 0.853, 0.874, 0.838, 0.919, 0.766],
    [0.921, 0.952, 0.804, 0.901, 0.948, 0.963, 0.948, 0.919, 0.953, 0.906, 0.97, 0.863],
    [0.765, 0.881, 0.674, 0.693, 0.85, 0.649, 0.689, 0.835, 0.657, 0.845, 0.851, 0.784]
)

yolo26s = Metrics(
    "yolo26s",
    [0.904, 0.96, 0.907, 0.846, 0.934, 0.926, 0.941, 0.907, 0.927, 0.853, 0.932, 0.814],
    [0.865, 0.878, 0.731, 0.857, 0.919, 0.942, 0.814, 0.88, 0.885, 0.855, 0.929, 0.82],
    [0.934, 0.956, 0.86, 0.902, 0.954, 0.976, 0.944, 0.946, 0.953, 0.912, 0.971, 0.901],
    [0.791, 0.896, 0.72, 0.74, 0.836, 0.718, 0.688, 0.876, 0.662, 0.861, 0.857, 0.843]
)

yolo26m = Metrics(
    "yolo26m",
    [0.921, 0.95, 0.918, 0.882, 0.942, 0.91, 0.924, 0.911, 0.933, 0.876, 0.946, 0.939],
    [0.885, 0.899, 0.787, 0.877, 0.923, 0.949, 0.875, 0.894, 0.901, 0.859, 0.943, 0.83],
    [0.944, 0.963, 0.882, 0.916, 0.957, 0.975, 0.959, 0.945, 0.961, 0.923, 0.976, 0.924],
    [0.827, 0.908, 0.739, 0.756, 0.891, 0.742, 0.788, 0.87, 0.755, 0.873, 0.906, 0.869]
)


def plot_xy_metric(
        destination: Path,
        metric_name: str,
        models_metric: list[Metrics],
        highlight: Metrics = None
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for model_results in models_metric:
        plt.plot(
            LABELS,
            model_results.get_metric(metric_name),
            marker='o',
            label=f"{model_results.model_size}",
            alpha=0.7 if highlight else 1.0
        )
    if highlight:
        plt.plot(
            LABELS,
            highlight.get_metric(metric_name),
            marker='o',
            label=f"Highlighted {highlight.model_size}",
            color='red',
            linewidth=2.5
        )

    plt.xlabel('Labels')
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.title(f'{metric_name.replace("_", " ").title()} per Label')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(destination / f"{metric_name}_per_label.png")
    plt.close()


def plot_metric_bars_percentage_improvement(
    destination: Path,
    metric_name: str,
    models_metric: list[Metrics],
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    num_labels = len(LABELS)
    num_models = len(models_metric)
    x = np.arange(num_labels)
    width = 0.8 / num_models  # Adjust bar width based on number of models

    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    # Get all metric values: list of lists, one inner list per model
    all_metrics = [model.get_metric(metric_name) for model in models_metric]

    # Find the worst performer (baseline) for each label
    baselines = np.min(all_metrics, axis=0)
    baseline_indices = np.argmin(all_metrics, axis=0)

    for i, model_results in enumerate(models_metric):
        metric_values = np.array(all_metrics[i])
        
        # Calculate percentage improvement over the dynamic baseline for each label
        improvements = np.where(baselines > 0, (metric_values - baselines) / baselines * 100, 0)
        
        # Set improvement to a small negative value for the baseline bar to indicate it
        # This bar will not be plotted, but the space is reserved.
        improvements[baseline_indices == i] = np.nan

        bar_positions = x - (width * num_models / 2) + (i + 0.5) * width
        ax.bar(bar_positions, improvements, width, label=f"{model_results.model_size}")

    plt.xlabel('Labels')
    plt.ylabel(f'Percentage Improvement in {metric_name.replace("_", " ").title()}')
    plt.title(f'Percentage Improvement vs. Worst Performer in {metric_name.replace("_", " ").title()} per Label')
    plt.xticks(x, LABELS, rotation=45, ha="right")
    plt.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    
    # Add a note about the baseline
    plt.figtext(0.5, 0.01, 'Note: For each label, the baseline (worst performer) is not shown. All other bars show % improvement over it.', 
        ha='center', fontsize=8, style='italic')

    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust layout to make space for figtext
    plt.savefig(destination / f"{metric_name}_percentage_improvement_per_label.png")
    plt.close()


# plot best yolo26n with s and m
for m in Metrics.available_metrics():
    plot_xy_metric(
        PLOTS_ROOT / "n_s_m_comparison", 
        m, 
        [yolo26n, yolo26s, yolo26m]
    )
    plot_metric_bars_percentage_improvement(
        PLOTS_ROOT / "n_s_m_comparison",
        m,
        [yolo26n, yolo26s, yolo26m]
    )

    