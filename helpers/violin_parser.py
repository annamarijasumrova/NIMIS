import re

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_score

def parse_assignment_file(path):
    predicted_slot = []
    true_slot = []
    conf = []
    n_components = []

    peak_line_pattern = re.compile(
        r"^t?[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s+ppm\s+(-?\d+)\s+(-?\d+)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$"
    )
    component_line_pattern = re.compile(r"^Component \d+:")

    current_n_components = None

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("Sample "):
                current_n_components = 0
                continue

            if component_line_pattern.match(line):
                current_n_components += 1
                continue

            match = peak_line_pattern.match(line)
            if match is None:
                continue

            pred = int(match.group(1))
            true = int(match.group(2))
            score = float(match.group(3))

            predicted_slot.append(pred)
            true_slot.append(true)
            conf.append(score)
            n_components.append(current_n_components)

    return predicted_slot, true_slot, conf, n_components

def plot_pr_curve(runs, title="Precision-Recall Curve (Peak-to-Component Assignment)",
                   save_path=None, show=False, ax=None):
    """runs: list of (label, predicted_slot, true_slot, conf) tuples, one PR curve each,
    overlaid on one axes -- e.g. one tuple per ablation run being compared."""
    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(6, 4.8))

    for label, predicted_slot, true_slot, conf in runs:
        y_true = [1 if predicted_slot[i] == true_slot[i] else 0 for i in range(len(conf))]
        precision, recall, _ = precision_recall_curve(y_true, conf)
        ap_value = average_precision_score(y_true, conf)
        ax.plot(recall, precision, label=f"{label} (AP={ap_value:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.75, 1.05)
    ax.set_title(title, fontweight='bold')
    ax.legend(loc="lower left")

    if owns_figure:
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        if show:
            plt.show()
        # plt.close(fig)


def plot_confidence_violin(conf_correct, conf_wrong, title="Peak-to-Component Assignment Confidence",
                            save_path=None, show=False, ax=None):
    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(7, 4.8))

    data = [conf_correct, conf_wrong]
    labels_plot = ["Correct", "Incorrect"]
    counts = [len(conf_correct), len(conf_wrong)]
    total_count = sum(counts)
    if total_count == 0:
        raise ValueError("plot_confidence_violin got no peaks (conf_correct and conf_wrong are both empty).")
    fractions = [c / total_count for c in counts]
    percentages = [100.0 * f for f in fractions]

    def summary(arr):
        arr = np.array(arr)
        if len(arr) == 0:
            return float("nan"), float("nan")
        mean = arr.mean()
        return (arr > mean).mean(), (arr < mean).mean()

    frac_above, frac_below = zip(*(summary(arr) for arr in data))

    n_above_correct = int(frac_above[0] * counts[0])
    n_below_wrong = int(frac_below[1] * counts[1])
    print(f"{title}: {n_above_correct}/{counts[0]} correct peaks above mean ({frac_above[0]*100:.1f}%), "
          f"{n_below_wrong}/{counts[1]} incorrect peaks below mean ({frac_below[1]*100:.1f}%)")

    max_width = 0.9
    min_width = 0.35
    max_fraction = max(fractions)
    widths = [max(min_width, max_width * (f / max_fraction)) for f in fractions]

    parts = ax.violinplot(
        [d if len(d) > 0 else [0.0] for d in data],
        positions=[1, 2],
        widths=widths,
        showmeans=True,
        showextrema=True,
        showmedians=False,
    )

    for pc in parts['bodies']:
        pc.set_facecolor('#FA8072')
        pc.set_alpha(0.45)
    for key in ['cmeans', 'cmaxes', 'cmins', 'cbars']:
        if key in parts:
            parts[key].set_color('#FA8072')

    xtick_labels = [
        f"{label}\n(Peak count {count}, {pct:.2f}%)"
        for label, count, pct in zip(labels_plot, counts, percentages)
    ]
    ax.set_xticks([1, 2], xtick_labels)
    ax.set_ylabel("NIMIS Score \nPeak-to-Component Assignment Confidence Measure", fontweight='bold')
    ax.set_title("Peak-to-Component Assignment Confidence", fontweight='bold')
    ax.set_ylim(top=1.08)

    if owns_figure:
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close(fig)


def plot_accuracy_by_size(n_components, predicted_slot, true_slot, conf,
                           title="Peak-to-Component Assignment Precision & Accuracy by Mixture Size",
                           save_path=None, show=False, ax=None):
    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(7, 4.8))

    n_components_arr = np.array(n_components)
    correct_arr = np.array([1 if predicted_slot[i] == true_slot[i] else 0 for i in range(len(conf))])
    conf_arr = np.array(conf)

    mixture_sizes = sorted(set(n_components))
    accuracy_by_size = []
    precision_by_size = []
    peak_counts_by_size = []

    for size in mixture_sizes:
        mask = n_components_arr == size
        y_true = correct_arr[mask]
        y_pred = (conf_arr[mask] >= 0.5).astype(int)

        accuracy_by_size.append(y_true.mean())
        precision_by_size.append(precision_score(y_true, y_pred, zero_division=0))
        peak_counts_by_size.append(mask.sum())

    x = np.arange(len(mixture_sizes))
    width = 0.35

    ax.bar(x - width / 2, accuracy_by_size, width, label="Accuracy", color='#FA8072')
    ax.bar(x + width / 2, precision_by_size, width, label="Precision (conf>=0.5)", color='#4682B4')
    ax.set_xticks(x, [f"{size}\n(n={count})" for size, count in zip(mixture_sizes, peak_counts_by_size)])
    ax.set_xlabel("Number of components in mixture")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_title(title, fontweight='bold')
    ax.legend(loc="lower right")

    if owns_figure:
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close(fig)


def plot_confidence_violin_grid(panels, save_path=None, show=False):
    """panels: list of up to 4 (title, conf_correct, conf_wrong) tuples, one per grid cell."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (title, conf_correct, conf_wrong) in zip(axes.flat, panels):
        plot_confidence_violin(conf_correct, conf_wrong, title=title, ax=ax)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_accuracy_by_size_grid(panels, save_path=None, show=False):
    """panels: list of up to 4 (title, n_components, predicted_slot, true_slot, conf) tuples."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (title, n_components, predicted_slot, true_slot, conf) in zip(axes.flat, panels):
        plot_accuracy_by_size(n_components, predicted_slot, true_slot, conf, title=title, ax=ax)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    data_path = "outputs/assignments_nmrexp_test_slot_conj_10.txt"

    pred, true, conf, n_comp = parse_assignment_file(data_path)
    conf_correct = [c for p_, t_, c in zip(pred, true, conf) if p_ == t_]
    conf_wrong = [c for p_, t_, c in zip(pred, true, conf) if p_ != t_]

    plot_pr_curve([(data_path, pred, true, conf)], show=True)
    plot_confidence_violin(conf_correct, conf_wrong, title=data_path, show=True)
    plot_accuracy_by_size(n_comp, pred, true, conf, title=data_path, show=True)
