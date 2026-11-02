"""Plot best/worst mixtures (by peak-slot assignment accuracy) as two combined
figures (best, worst). Each figure linearly stacks one section per component
count (3, 4, 5), each section titled by its component count and containing
3 mixture rows (structures + fraction labels). Every mixture row is tagged
with a continuous letter (A, B, C, ...) across the whole figure and its
accuracy, separated by horizontal rules.
"""
import re
import string
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

ASSIGNMENTS_FILE = Path(__file__).resolve().parent.parent / "outputs" / "assignments_nmrexp_val_slot_conj_10_molecule.txt"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "case_studies"
STRUCT_SIZE = 300

N_EXAMPLES = 3
COMPONENT_COUNTS = (3, 4, 5)

TOP_MARGIN = 0.6
BOTTOM_MARGIN = 0.1
TITLE_H = 0.5
ROW_H = 2.7
SECTION_GAP = 0.4
LEFT_FRAC = 0.09
RIGHT_FRAC = 0.97
FIG_WIDTH = 13


def parse_samples(path):
    samples = []
    cur = None
    batch = None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("=== Batch"):
                m = re.search(r"Batch (\d+)/", line)
                if m:
                    batch = int(m.group(1))
            if line.startswith("Sample "):
                if cur is not None:
                    samples.append(cur)
                cur = {"id": line, "batch": batch, "smiles": None, "fractions": [], "peaks": []}
            elif line.startswith("SMILES:"):
                cur["smiles"] = [s.strip() for s in line[len("SMILES:"):].split(",")]
            elif line.startswith("  Component "):
                frac = float(line.rsplit("=", 1)[1])
                cur["fractions"].append(frac)
            elif line.startswith("t") and " ppm" in line and cur is not None:
                parts = line.split("\t")
                if len(parts) == 4:
                    try:
                        cur["peaks"].append((int(parts[1]), int(parts[2])))
                    except ValueError:
                        pass
        if cur is not None:
            samples.append(cur)

    for s in samples:
        s["n_components"] = len(s["smiles"])
        n = len(s["peaks"])
        correct = sum(1 for p, t in s["peaks"] if p == t)
        s["n_peaks"] = n
        s["acc"] = correct / n if n else None
    return samples


def best_and_worst(samples, n_comp, n_each=N_EXAMPLES):
    group = [s for s in samples if s["n_components"] == n_comp and s["n_peaks"] > 0]
    group.sort(key=lambda s: (-s["acc"], -s["n_peaks"]))
    return group[:n_each], group[-n_each:]


def plot_combined(groups, title, out_path):
    n_sections = len(groups)

    bounds = []
    cum = TOP_MARGIN
    for n_comp, mixtures in groups:
        title_top = cum
        title_bottom = title_top + TITLE_H
        rows_top = title_bottom
        rows_bottom = rows_top + len(mixtures) * ROW_H
        bounds.append((n_comp, mixtures, title_top, title_bottom, rows_top, rows_bottom))
        cum = rows_bottom + SECTION_GAP
    total_height = cum - SECTION_GAP + BOTTOM_MARGIN

    def to_frac(y_in):
        return 1 - y_in / total_height

    fig = plt.figure(figsize=(FIG_WIDTH, total_height))
    letters = iter(string.ascii_uppercase)

    for section_idx, (n_comp, mixtures, title_top, title_bottom, rows_top, rows_bottom) in enumerate(bounds):
        if section_idx > 0:
            y = to_frac(title_top)
            fig.add_artist(plt.Line2D([0.02, 0.98], [y, y], transform=fig.transFigure,
                                       color="black", linewidth=1.2))

        fig.text(0.5, to_frac((title_top + title_bottom) / 2), f"{n_comp}-component mixtures",
                  ha="center", va="center", fontsize=13, fontweight="bold")

        col_w = (RIGHT_FRAC - LEFT_FRAC) / n_comp
        pad = col_w * 0.04

        for r, mix in enumerate(mixtures):
            letter = next(letters)
            row_top_in = rows_top + r * ROW_H
            row_bottom_in = row_top_in + ROW_H
            top_frac = to_frac(row_top_in)
            bottom_frac = to_frac(row_bottom_in)

            fig.text(LEFT_FRAC - 0.045, (top_frac + bottom_frac) / 2,
                      f"{letter}\nAccuracy = {mix['acc']:.2f}",
                      rotation=90, ha="center", va="center", fontsize=9, fontweight="bold")

            for c in range(n_comp):
                left = LEFT_FRAC + c * col_w + pad
                width = col_w - 2 * pad
                ax = fig.add_axes([left, bottom_frac, width, top_frac - bottom_frac])
                ax.axis("off")
                smi = mix["smiles"][c]
                frac = mix["fractions"][c]
                mol = Chem.MolFromSmiles(smi)
                img = Draw.MolToImage(mol, size=(STRUCT_SIZE, STRUCT_SIZE))
                ax.imshow(img)
                ax.set_title(f"fraction = {frac:.3f}", fontsize=8, pad=3)

            is_last_row_overall = section_idx == n_sections - 1 and r == len(mixtures) - 1
            if not is_last_row_overall:
                fig.add_artist(plt.Line2D([0.02, 0.98], [bottom_frac, bottom_frac], transform=fig.transFigure,
                                           color="lightgray", linewidth=0.8))

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1 - (TOP_MARGIN * 0.4) / total_height)
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    samples = parse_samples(ASSIGNMENTS_FILE)
    best_groups, worst_groups = [], []
    for n_comp in COMPONENT_COUNTS:
        best, worst = best_and_worst(samples, n_comp)
        best_groups.append((n_comp, best))
        worst_groups.append((n_comp, worst))

    plot_combined(best_groups, "Best peak-to-component assignment accuracy",
                  OUT_DIR / "mixtures_best.png")
    plot_combined(worst_groups, "Worst peak-to-component assignment accuracy",
                  OUT_DIR / "mixtures_worst.png")


if __name__ == "__main__":
    main()
