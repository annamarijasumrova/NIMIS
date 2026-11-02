"""Peak-to-component assignment accuracy broken down by the true component's
concentration in the mixture (1-5%, 5-10%, 10-15%, 15-20%, >20%).

Rare/minor components are the practically hard case for mixture deconvolution
(weak peaks, easy to misassign) and also the case `con` (the intensity-ratio
bias) was nominally designed to help with, since a component's peaks share
its scale factor. This checks both: how accuracy varies with concentration,
and whether con/conJ's contribution concentrates in the low-concentration
brackets the way the overlap-bucket check (stratified_test_conj.py) ruled
out for peak overlap.

No test-parquet join needed here, unlike stratified_test_conj.py: each
assignment .txt file already logs "Component N: SMILES  proportion=X" per
sample, so a peak's true-component concentration is read directly out of the
same file as its prediction -- no row-alignment/batch_size bookkeeping.
"""
import argparse
import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

SAMPLE_RE = re.compile(r"^Sample \d+$")
COMPONENT_RE = re.compile(r"^Component (\d+): .*proportion=([-+]?\d*\.?\d+)$")
PEAK_RE = re.compile(
    r"^t?[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s+ppm\s+(-?\d+)\s+(-?\d+)\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$"
)

BRACKETS = [(0.01, 0.05, "1-5%"), (0.05, 0.10, "5-10%"), (0.10, 0.15, "10-15%"),
            (0.15, 0.20, "15-20%"), (0.20, np.inf, ">20%")]


def parse_assignments_with_concentration(path):
    """Returns (pred, true, concentration, n_components) arrays, one entry per peak.
    n_components is the number of components in that peak's own mixture (len of
    the Component-proportions block for the sample the peak came from)."""
    pred, true, conc, ncomp = [], [], [], []
    proportions = {}
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if SAMPLE_RE.match(line):
                proportions = {}
                continue
            m = COMPONENT_RE.match(line)
            if m:
                proportions[int(m.group(1))] = float(m.group(2))
                continue
            m = PEAK_RE.match(line)
            if m:
                p, t = int(m.group(1)), int(m.group(2))
                if t in proportions:
                    pred.append(p)
                    true.append(t)
                    conc.append(proportions[t])
                    ncomp.append(len(proportions))
    return np.array(pred), np.array(true), np.array(conc), np.array(ncomp)


def accuracy_by_bracket(pred, true, conc):
    correct = pred == true
    out = []
    for lo, hi, label in BRACKETS:
        mask = (conc >= lo) & (conc < hi)
        n = mask.sum()
        acc = correct[mask].mean() if n else float("nan")
        out.append((label, n, acc))
    return out


def accuracy_by_bracket_and_ncomp(pred, true, conc, ncomp, ncomp_values):
    """{n_components: [(bracket_label, n, acc), ...]}, same brackets as accuracy_by_bracket."""
    correct = pred == true
    out = {}
    for c in ncomp_values:
        rows = []
        c_mask = ncomp == c
        for lo, hi, label in BRACKETS:
            mask = c_mask & (conc >= lo) & (conc < hi)
            n = mask.sum()
            acc = correct[mask].mean() if n else float("nan")
            rows.append((label, n, acc))
        out[c] = rows
    return out


def binned_accuracy_by_ncomp(pred, true, conc, ncomp, ncomp_values, bin_width_pct=2.0):
    """Fine-grained (bin_width_pct wide) accuracy bins spanning the full 0-100%
    concentration grid, per n_components. Returns {n_components: (bin_centers,
    accuracy_pct, n_per_bin)} -- accuracy_pct/n_per_bin are NaN/0 for empty bins,
    so every group shares the same x grid."""
    correct = pred == true
    conc_pct = conc * 100
    edges = np.arange(0, 100 + bin_width_pct, bin_width_pct)
    centers = (edges[:-1] + edges[1:]) / 2

    out = {}
    for c in ncomp_values:
        c_mask = ncomp == c
        ys = np.full(len(centers), np.nan)
        ns = np.zeros(len(centers))
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            m = c_mask & (conc_pct >= lo) & (conc_pct < hi)
            n = m.sum()
            if n > 0:
                ys[i] = correct[m].mean() * 100
                ns[i] = n
        out[c] = (centers, ys, ns)
    return out


def rolling_trend(ys, ns, window_bins):
    """Weighted (by bin count) moving average over ys, ignoring empty (NaN) bins
    inside the window. NaN where the window has no data at all."""
    n_bins = len(ys)
    half = window_bins // 2
    trend = np.full(n_bins, np.nan)
    for i in range(n_bins):
        lo, hi = max(0, i - half), min(n_bins, i + half + 1)
        w, y = ns[lo:hi], ys[lo:hi]
        valid = ~np.isnan(y) & (w > 0)
        if valid.any():
            trend[i] = np.average(y[valid], weights=w[valid])
    return trend


def plot_accuracy_by_concentration_and_ncomp(pred, true, conc, ncomp, ncomp_values, title, save_path,
                                              bin_width_pct=2.0, trend_window_bins=7, y_min=0.4):
    """One weighted-moving-average trendline per n_components, x axis spanning
    the full 0-100% concentration range, y axis in 0-1 fraction units. The
    trendline is a local average of real accuracy values (bounded 0-1, no
    polynomial overshoot) restricted to each group's own observed
    concentration range. Prints (does not plot) the concentration at which
    each group's trendline is lowest."""
    import matplotlib.pyplot as plt
    binned = binned_accuracy_by_ncomp(pred, true, conc, ncomp, ncomp_values, bin_width_pct)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    fig.patch.set_facecolor("white")
    ax.patch.set_facecolor("white")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    minima = {}
    for c, color in zip(ncomp_values, colors):
        centers, ys, ns = binned[c]
        has_data = ns > 0
        if not has_data.any():
            continue

        trend = rolling_trend(ys, ns, trend_window_bins)
        first, last = np.flatnonzero(has_data)[[0, -1]]
        trend_mask = ~np.isnan(trend) & (np.arange(len(trend)) >= first) & (np.arange(len(trend)) <= last)
        x_trend, y_trend = centers[trend_mask], trend[trend_mask] / 100.0
        ax.plot(x_trend, y_trend, color=color, linewidth=2.5, label=f"{c}-component mixture")

        worst = np.argmin(y_trend)
        minima[c] = (x_trend[worst], y_trend[worst])

    ax.set_xlim(0, 100)
    ax.set_ylim(y_min, 1.0)
    ax.set_xlabel("Component Fraction in the Mixture (%)")
    ax.set_ylabel("NIMIS Score\nPeak-to-Component Assignment Accuracy")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, facecolor="white")

    print("lowest assignment accuracy per mixture type:")
    for c in ncomp_values:
        if c in minima:
            frac_pct, acc = minima[c]
            print(f"  {c}-component mixture: min accuracy {acc:.3f} at component fraction {frac_pct:.1f}%")
    return binned, minima


def demo():
    """Self-check: 2 samples, hand-picked proportions, verify bracket accuracy."""
    import tempfile
    text = """Peak assignments

Sample 0
SMILES: A, B
Component proportions:
  Component 0: A\tproportion=0.0300
  Component 1: B\tproportion=0.9700
peak_position_ppm\tpred_mapped_slot\ttrue_slot\tconfidence
t1.0 ppm\t0\t0\t0.90
t2.0 ppm\t1\t0\t0.60
t3.0 ppm\t1\t1\t0.95

Sample 1
SMILES: C, D
Component proportions:
  Component 0: C\tproportion=0.1200
  Component 1: D\tproportion=0.8800
peak_position_ppm\tpred_mapped_slot\ttrue_slot\tconfidence
t4.0 ppm\t0\t0\t0.85
"""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(text)
        fpath = f.name

    pred, true, conc, ncomp = parse_assignments_with_concentration(fpath)
    assert pred.tolist() == [0, 1, 1, 0]
    assert true.tolist() == [0, 0, 1, 0]
    assert conc.tolist() == [0.03, 0.03, 0.97, 0.12]
    assert ncomp.tolist() == [2, 2, 2, 2]

    result = accuracy_by_bracket(pred, true, conc)
    by_label = {label: (n, acc) for label, n, acc in result}
    assert by_label["1-5%"] == (2, 0.5)
    assert by_label["10-15%"] == (1, 1.0)
    assert by_label[">20%"] == (1, 1.0)

    by_ncomp = accuracy_by_bracket_and_ncomp(pred, true, conc, ncomp, ncomp_values=(2, 3))
    by_label_2 = {label: (n, acc) for label, n, acc in by_ncomp[2]}
    assert by_label_2["1-5%"] == (2, 0.5)
    assert all(n == 0 for _, n, _ in by_ncomp[3])

    Path(fpath).unlink()
    print("concentration_accuracy self-check passed")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", default=[], metavar="LABEL=PATH", help="add/override a run; repeatable")
    p.add_argument("--plot-out", type=Path, default=None,
                    help="save an accuracy-vs-concentration plot for --plot-run, one line per --ncomp-values")
    p.add_argument("--plot-run", default="Drop con", help="which run label to plot (default: Drop con, i.e. run 10)")
    p.add_argument("--ncomp-values", type=str, default="2,3,4,5")
    p.add_argument("--bin-width-pct", type=float, default=2.0, help="width of the scatter bins in concentration %%")
    p.add_argument("--trend-window-bins", type=int, default=7, help="moving-average window (in bins) for the trendline")
    args = p.parse_args()

    runs = {
        "Full (con+conJ)": REPO_ROOT / "outputs/assignments_nmrexp_test_slot_conj_8.txt",
        "Vanilla":         REPO_ROOT / "outputs/assignments_nmrexp_test_slot_conj_9.txt",
        "Drop con":        REPO_ROOT / "outputs/assignments_nmrexp_test_slot_conj_10.txt",
        "Drop conJ":       REPO_ROOT / "outputs/assignments_nmrexp_test_slot_conj_11.txt",
    }
    for spec in args.run:
        label, path = spec.split("=", 1)
        runs[label] = Path(path)

    parsed = {label: parse_assignments_with_concentration(path) for label, path in runs.items()}
    per_run = {label: accuracy_by_bracket(*p_[:3]) for label, p_ in parsed.items()}

    if args.plot_out is not None:
        ncomp_values = tuple(int(c) for c in args.ncomp_values.split(","))
        pred, true, conc, ncomp = parsed[args.plot_run]
        plot_accuracy_by_concentration_and_ncomp(
            pred, true, conc, ncomp, ncomp_values,
            title="NIMIS Peak-to-Component Assignment Accuracy vs. Component Fraction",
            save_path=args.plot_out,
            bin_width_pct=args.bin_width_pct, trend_window_bins=args.trend_window_bins,
        )
        print(f"wrote {args.plot_out}")

    n_brackets = len(BRACKETS)
    header = f"{'concentration':>14}{'n peaks':>10}" + "".join(f"{l:>20}" for l in runs)
    has_full = "Full (con+conJ)" in runs
    if has_full and "Drop con" in runs:
        header += f"{'con gain':>11}"
    if has_full and "Drop conJ" in runs:
        header += f"{'conJ gain':>11}"
    print(header)
    for i in range(n_brackets):
        label, n, _ = per_run[next(iter(runs))][i]
        row = f"{label:>14}{n:>10}"
        for run_label in runs:
            _, _, acc = per_run[run_label][i]
            row += f"{acc*100:>19.2f}%"
        if has_full and "Drop con" in runs:
            row += f"{(per_run['Full (con+conJ)'][i][2] - per_run['Drop con'][i][2])*100:>10.2f}pt"
        if has_full and "Drop conJ" in runs:
            row += f"{(per_run['Full (con+conJ)'][i][2] - per_run['Drop conJ'][i][2])*100:>10.2f}pt"
        print(row)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--selfcheck":
        demo()
    else:
        main()
