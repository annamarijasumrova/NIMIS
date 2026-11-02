"""Find an operating threshold on the NIMIS score (slot-attention confidence)
above which a peak-to-component assignment should be trusted as correct.

Treats "is this assignment correct" as a binary classification problem with
the confidence score as the classifier: reuses violin_parser's own
parse_assignment_file (correct = pred == true) and finds:

  - the Youden's-J-optimal threshold from the ROC curve (maximizes
    TPR - FPR -- the standard "best single cutpoint" for separating two
    score distributions, without picking an arbitrary target precision)
  - the F1-optimal threshold from the precision-recall curve
  - precision/recall/accuracy/coverage at both, plus a spread of round
    thresholds, so the trade-off is visible instead of hidden behind one
    number

"Coverage" = fraction of all peaks that clear the threshold -- matters
because a high threshold can be precise while discarding most predictions.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from helpers.violin_parser import parse_assignment_file

REPO_ROOT = Path(__file__).resolve().parent.parent


def youden_optimal_threshold(y_true, scores):
    fpr, tpr, thresh = roc_curve(y_true, scores)
    j = tpr - fpr
    idx = np.argmax(j)
    return float(thresh[idx])


def f1_optimal_threshold(y_true, scores):
    precision, recall, thresh = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    idx = np.argmax(f1[:-1])
    return float(thresh[idx])


def accuracy_optimal_threshold(y_true, scores):
    """Threshold that maximizes the raw accuracy of the "score >= t means
    correct" call. Differs from Youden's J when classes are imbalanced (here,
    ~74% of peaks are correctly assigned): Youden's J balances TPR/FPR
    regardless of class prevalence, so on an imbalanced set it can pick a
    threshold with *worse* raw accuracy than just calling everything
    correct. This is the criterion that actually matches "assignments above
    t should be deemed correct, below incorrect.\""""
    order = np.argsort(-scores, kind="stable")
    y_sorted = y_true[order]
    s_sorted = scores[order]
    n = len(y_true)
    total_neg = n - y_true.sum()

    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.arange(1, n + 1) - tp_cum
    tn_cum = total_neg - fp_cum
    acc = (tp_cum + tn_cum) / n

    is_boundary = np.empty(n, dtype=bool)
    is_boundary[:-1] = s_sorted[:-1] != s_sorted[1:]
    is_boundary[-1] = True
    acc_valid = np.where(is_boundary, acc, -1.0)

    idx = int(np.argmax(acc_valid))
    return float(s_sorted[idx])


def metrics_at_threshold(y_true, scores, t):
    kept = scores >= t
    n_total = len(scores)
    n_kept = int(kept.sum())
    coverage = n_kept / n_total if n_total else float("nan")
    precision = y_true[kept].mean() if n_kept else float("nan")
    recall = (y_true & kept).sum() / y_true.sum() if y_true.sum() else float("nan")
    accuracy = (kept == y_true).mean()
    return {"threshold": t, "coverage": coverage, "precision": precision,
            "recall": recall, "accuracy": accuracy}


def demo():
    """Self-check: cleanly separated scores should push both optimal
    thresholds into the gap between the two clusters."""
    rng = np.random.default_rng(0)
    correct_scores = rng.uniform(0.85, 1.0, size=200)
    wrong_scores = rng.uniform(0.0, 0.4, size=50)
    scores = np.concatenate([correct_scores, wrong_scores])
    y_true = np.concatenate([np.ones(200, dtype=bool), np.zeros(50, dtype=bool)])

    gap_lo, gap_hi = wrong_scores.max(), correct_scores.min()
    yj = youden_optimal_threshold(y_true, scores)
    f1t = f1_optimal_threshold(y_true, scores)
    acct = accuracy_optimal_threshold(y_true, scores)
    assert gap_lo < yj <= gap_hi, yj
    assert gap_lo < f1t <= gap_hi, f1t
    assert gap_lo < acct <= gap_hi, acct

    m = metrics_at_threshold(y_true, scores, yj)
    assert m["precision"] > 0.99 and m["recall"] > 0.99

    tie_scores = np.array([0.9, 0.9, 0.9, 0.2, 0.2])
    tie_labels = np.array([True, True, False, False, False])
    acct_tie = accuracy_optimal_threshold(tie_labels, tie_scores)
    assert acct_tie in (0.9, 0.2), acct_tie

    print("nimis_score_threshold self-check passed")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assignments", type=Path,
                    default=REPO_ROOT / "outputs/assignments_nmrexp_test_slot_conj_10.txt",
                    help="default: run 10, drop-con -- the recommended model")
    p.add_argument("--extra-thresholds", type=str, default="0.5,0.7,0.8,0.9,0.95,0.99")
    args = p.parse_args()

    pred, true, conf, _ = parse_assignment_file(args.assignments)
    y_true = np.array(pred) == np.array(true)
    scores = np.array(conf)

    print(f"{len(scores):,} peaks | base rate (fraction correct overall): {y_true.mean()*100:.2f}%\n")

    yj = youden_optimal_threshold(y_true, scores)
    f1t = f1_optimal_threshold(y_true, scores)
    acct = accuracy_optimal_threshold(y_true, scores)

    rows = [("Accuracy optimal", acct), ("Youden's J optimal", yj), ("F1 optimal", f1t)]
    rows += [(f"fixed t={t}", t) for t in (float(x) for x in args.extra_thresholds.split(","))]

    print(f"{'threshold source':>22}{'NIMIS score':>13}{'coverage':>11}{'precision':>12}{'recall':>10}{'accuracy':>11}")
    for label, t in rows:
        m = metrics_at_threshold(y_true, scores, t)
        print(f"{label:>22}{m['threshold']:>13.4f}{m['coverage']*100:>10.2f}%"
              f"{m['precision']*100:>11.2f}%{m['recall']*100:>9.2f}%{m['accuracy']*100:>10.2f}%")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selfcheck":
        demo()
    else:
        main()
