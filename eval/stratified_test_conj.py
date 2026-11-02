"""How much does conJ (J-coupling attention bias) help, broken down by how badly
peaks overlap in a mixture -- instead of one aggregate accuracy number.

conJ was designed to disambiguate peaks that overlap in chemical shift (shift
alone can't tell them apart, but their coupling constants can). The aggregate
test accuracy only showed a flat ~1.5pt gain over vanilla; this checks whether
that gain is actually concentrated in the high-overlap mixtures it was meant
for, using the `max_overlap` column already computed by
new_dataset/all_mix_for_original_nimis.py (0 = no cross-component peak overlap
in that mixture, up to 1 = full overlap).

How: the assignment .txt files (written by evaluate_loader in
scripts/train.py) list predictions per "Sample" block within
"Batch" blocks, but helpers.violin_parser.parse_assignment_file flattens that away.
The test DataLoader uses shuffle=False, so the original test_df row order is
recoverable as `global_row = batch_idx * batch_size + sample_idx_in_batch`.
Once each block is mapped back to its row, we can pull that row's
`max_overlap` and bucket peaks by it. Alignment is verified against the
SMILES list logged in each block -- if batch_size is wrong for a file, this
mismatches immediately instead of silently scrambling the buckets.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

BATCH_RE = re.compile(r"^=== Batch (\d+)/\d+ ===$")
SAMPLE_RE = re.compile(r"^Sample (\d+)$")
PEAK_RE = re.compile(
    r"^t?[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s+ppm\s+(-?\d+)\s+(-?\d+)\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$"
)


def parse_assignments_with_rows(path, batch_size):
    """Returns list of (global_row, smiles_list, [(pred, true), ...]) per sample block."""
    results = []
    batch_idx = sample_idx = smiles = None
    peaks = []

    def flush():
        if sample_idx is not None:
            results.append((batch_idx * batch_size + sample_idx, smiles, peaks[:]))

    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            m = BATCH_RE.match(line)
            if m:
                flush()
                batch_idx, sample_idx, smiles, peaks = int(m.group(1)) - 1, None, None, []
                continue
            m = SAMPLE_RE.match(line)
            if m:
                flush()
                sample_idx, smiles, peaks = int(m.group(1)), None, []
                continue
            if line.startswith("SMILES: "):
                smiles = line[len("SMILES: "):].split(", ")
                continue
            m = PEAK_RE.match(line)
            if m:
                peaks.append((int(m.group(1)), int(m.group(2))))
        flush()
    return results


def load_test_df(path):
    """Mirrors load_multi() in scripts/train.py -- must match exactly,
    since that filter+reindex is what defines the row order the assignment
    files were written against."""
    df = pd.read_parquet(path)
    return df[df["n_components"] >= 2].reset_index(drop=True)


def peaks_and_overlap_for_run(path, batch_size, test_df):
    parsed = parse_assignments_with_rows(path, batch_size)
    rows = np.array([r for r, _, _ in parsed])
    mismatches = sum(
        1 for r, smiles, _ in parsed if list(test_df.iloc[r]["smiles"]) != smiles
    )
    if mismatches:
        raise RuntimeError(
            f"{path}: {mismatches}/{len(parsed)} sample blocks don't match test_df at "
            f"the row batch_size={batch_size} implies -- wrong --batch-size for this run?"
        )

    correct, overlap = [], []
    max_overlap = test_df["max_overlap"].to_numpy()
    for row, _, peaks in parsed:
        for pred, true in peaks:
            correct.append(pred == true)
            overlap.append(max_overlap[row])
    return np.array(correct), np.array(overlap)


def bucket_edges(overlap, n_buckets):
    """Zero-overlap peaks get their own bucket (no cross-component crowding at
    all); the rest are split into n_buckets equal-count quantile bins."""
    nonzero = overlap[overlap > 0]
    _, edges = pd.qcut(nonzero, q=n_buckets, retbins=True, duplicates="drop")
    return edges


def bucket_labels(edges):
    if np.unique(edges).size == 1:
        return ["no overlap", f"{edges[0]:.2f}"]
    return ["no overlap"] + [f"({edges[i]:.2f}, {edges[i+1]:.2f}]" for i in range(len(edges) - 1)]


def assign_buckets(overlap, edges):
    """bucket 0 = exactly zero overlap; bucket b>0 = nonzero quantile bin b-1."""
    idx = np.zeros(len(overlap), dtype=int)
    nonzero_mask = overlap > 0
    if not nonzero_mask.any():
        return idx
    if np.unique(edges).size == 1:
        idx[nonzero_mask] = 1
        return idx
    clipped = np.clip(overlap[nonzero_mask], edges[0], edges[-1])
    codes = pd.cut(clipped, bins=edges, labels=False, include_lowest=True)
    idx[nonzero_mask] = codes + 1
    return idx


def accuracy_per_bucket(correct, overlap, edges):
    labels = bucket_labels(edges)
    bucket_idx = assign_buckets(overlap, edges)
    out = []
    for b, label in enumerate(labels):
        mask = bucket_idx == b
        n = mask.sum()
        acc = correct[mask].mean() if n else float("nan")
        out.append((label, n, acc))
    return out


def demo():
    """Self-check: fake 2-batch/batch_size=2 assignment file + matching test_df,
    verify row recovery and bucketing land where expected."""
    import tempfile

    text = """Peak assignments

=== Batch 1/2 ===

Sample 0
Solvent: CDCl3
Frequency: 500.0
SMILES: A, B
Component proportions:
  Component 0: A\tproportion=0.5
  Component 1: B\tproportion=0.5
peak_position_ppm\tpred_mapped_slot\ttrue_slot\tconfidence
t1.0 ppm\t0\t0\t0.99
t2.0 ppm\t1\t0\t0.60

Sample 1
Solvent: CDCl3
Frequency: 500.0
SMILES: C
Component proportions:
  Component 0: C\tproportion=1.0
peak_position_ppm\tpred_mapped_slot\ttrue_slot\tconfidence
t3.0 ppm\t0\t0\t0.99

=== Batch 2/2 ===

Sample 0
Solvent: CDCl3
Frequency: 500.0
SMILES: D, E
Component proportions:
  Component 0: D\tproportion=0.5
  Component 1: E\tproportion=0.5
peak_position_ppm\tpred_mapped_slot\ttrue_slot\tconfidence
t4.0 ppm\t0\t1\t0.55
"""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(text)
        fpath = f.name

    test_df = pd.DataFrame({
        "smiles": [["A", "B"], ["C"], ["D", "E"]],
        "max_overlap": [0.0, 0.9, 0.9],
    })

    correct, overlap = peaks_and_overlap_for_run(fpath, batch_size=2, test_df=test_df)
    assert correct.tolist() == [True, False, True, False]
    assert overlap.tolist() == [0.0, 0.0, 0.9, 0.9]

    edges = bucket_edges(overlap, n_buckets=1)
    result = accuracy_per_bucket(correct, overlap, edges)
    assert result[0] == ("no overlap", 2, 0.5)
    assert result[1][1] == 2 and result[1][2] == 0.5

    Path(fpath).unlink()
    print("stratified_test_conj self-check passed")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-parquet", type=Path, default=REPO_ROOT / "data/test_molecule_up25_orig_1prct.parquet")
    p.add_argument("--n-buckets", type=int, default=4, help="quantile buckets for mixtures with overlap > 0")
    p.add_argument("--run", action="append", default=[], metavar="LABEL=PATH:BATCH_SIZE",
                    help="add/override a run; repeatable")
    args = p.parse_args()

    runs = {
        "Full (con+conJ)": (REPO_ROOT / "outputs/assignments_nmrexp_test_slot_conj_8.txt", 32),
        "Vanilla":         (REPO_ROOT / "outputs/assignments_nmrexp_test_slot_conj_9.txt", 32),
        "Drop con":        (REPO_ROOT / "outputs/assignments_nmrexp_test_slot_conj_10.txt", 32),
        "Drop conJ":       (REPO_ROOT / "outputs/assignments_nmrexp_test_slot_conj_11.txt", 64),
    }
    for spec in args.run:
        label, rest = spec.split("=", 1)
        path, bs = rest.rsplit(":", 1)
        runs[label] = (Path(path), int(bs))

    test_df = load_test_df(args.test_parquet)

    per_run = {}
    for label, (path, bs) in runs.items():
        correct, overlap = peaks_and_overlap_for_run(path, bs, test_df)
        per_run[label] = (correct, overlap)

    edges = bucket_edges(next(iter(per_run.values()))[1], args.n_buckets)
    per_run_buckets = {label: accuracy_per_bucket(c, o, edges) for label, (c, o) in per_run.items()}

    n_bins = len(next(iter(per_run_buckets.values())))
    header = f"{'overlap bucket':>16}" + "".join(f"{l:>20}" for l in runs)
    if "Full (con+conJ)" in runs and "Drop conJ" in runs:
        header += f"{'conJ gain':>12}"
    print(header)
    for i in range(n_bins):
        label = per_run_buckets[next(iter(runs))][i][0]
        n = per_run_buckets[next(iter(runs))][i][1]
        row = f"{label:>16}"
        for run_label in runs:
            _, _, acc = per_run_buckets[run_label][i]
            row += f"{acc*100:>19.2f}%"
        if "Full (con+conJ)" in runs and "Drop conJ" in runs:
            full_acc = per_run_buckets["Full (con+conJ)"][i][2]
            dropconj_acc = per_run_buckets["Drop conJ"][i][2]
            row += f"{(full_acc - dropconj_acc)*100:>11.2f}pt"
        print(row, f"  (n={n})")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selfcheck":
        demo()
    else:
        main()
