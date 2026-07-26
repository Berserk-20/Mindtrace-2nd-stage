"""
compare_results.py — MindTrace FER Research
============================================
Reads results/metrics_log.json and prints a formatted comparison table
suitable for copy-pasting into a research paper (also outputs LaTeX).

Usage
-----
    python compare_results.py                        # all results
    python compare_results.py --dataset rafdb        # filter by dataset
    python compare_results.py --sort f1              # sort by macro-F1
    python compare_results.py --latex                # show LaTeX table only
"""

import json
import os
import argparse

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "metrics_log.json")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="MindTrace FER — Model Comparison Table",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Filter results by dataset (rafdb | fer2013 | ckplus)",
    )
    parser.add_argument(
        "--sort", default="val_acc",
        choices=["val_acc", "f1", "params", "inference", "training_time"],
        help="Column to sort by",
    )
    parser.add_argument(
        "--latex", action="store_true",
        help="Print LaTeX table only",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────
# SORTING
# ──────────────────────────────────────────────────────────────────────
SORT_KEYS = {
    "val_acc":       lambda r: r["best_val_acc"],
    "f1":            lambda r: r["best_val_f1"],
    "params":        lambda r: r["params_total_M"],
    "inference":     lambda r: r["inference_time_ms"],
    "training_time": lambda r: r["training_time_sec"],
}
# Higher is better for acc/f1; lower is better for params/inference/time
SORT_DESC = {"val_acc": True, "f1": True,
             "params": False, "inference": False, "training_time": False}


# ──────────────────────────────────────────────────────────────────────
# CONSOLE TABLE
# ──────────────────────────────────────────────────────────────────────
def print_console_table(runs, sort_by):
    col_widths = [14, 10, 10, 10, 10, 11, 12]
    headers    = [
        "Model", "Dataset",
        "Val Acc%", "Macro F1%",
        "Params(M)", "Infer(ms)", "Train(min)",
    ]
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_widths) + " |"

    print(f"\n{'='*90}")
    print(f"  MindTrace FER — Model Comparison   (sorted by {sort_by})")
    print(f"{'='*90}")
    print(sep)
    print(fmt.format(*headers))
    print(sep)

    for r in runs:
        print(fmt.format(
            r["model"],
            r["dataset"],
            f"{r['best_val_acc']:.2f}",
            f"{r['best_val_f1']:.2f}",
            f"{r['params_total_M']:.2f}",
            f"{r['inference_time_ms']:.2f}",
            f"{r['training_time_sec'] / 60:.1f}",
        ))
    print(sep)
    print()


# ──────────────────────────────────────────────────────────────────────
# LATEX TABLE
# ──────────────────────────────────────────────────────────────────────
def print_latex_table(runs):
    print("\n% ── LaTeX Table (paste into your paper) ────────────────")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Comparison of Emotion Recognition Models on FER Benchmarks}")
    print("\\label{tab:fer_comparison}")
    print("\\begin{tabular}{llrrrrrr}")
    print("\\hline")
    print(
        "\\textbf{Model} & \\textbf{Dataset} & "
        "\\textbf{Val Acc (\\%)} & \\textbf{Macro F1 (\\%)} & "
        "\\textbf{Params (M)} & \\textbf{Infer (ms)} & "
        "\\textbf{Train (min)} \\\\"
    )
    print("\\hline")

    prev_dataset = None
    for r in runs:
        # Add a visual separator between datasets
        if prev_dataset and r["dataset"] != prev_dataset:
            print("\\hline")
        prev_dataset = r["dataset"]

        print(
            f"{r['model']} & {r['dataset']} & "
            f"{r['best_val_acc']:.2f} & {r['best_val_f1']:.2f} & "
            f"{r['params_total_M']:.2f} & {r['inference_time_ms']:.2f} & "
            f"{r['training_time_sec']/60:.1f} \\\\"
        )

    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    print()


# ──────────────────────────────────────────────────────────────────────
# PER-CLASS BREAKDOWN
# ──────────────────────────────────────────────────────────────────────
def print_per_class_breakdown(runs):
    """Print per-emotion F1 for each run (useful for ablation tables)."""
    print(f"\n{'='*90}")
    print("  Per-Class F1 Score Breakdown (Best Model per run)")
    print(f"{'='*90}")

    for r in runs:
        print(f"\n  {r['model'].upper()} on {r['dataset'].upper()}")
        report = r.get("classification_report", {})
        class_names = r.get("class_names", [])
        if not report or not class_names:
            print("    (no per-class data)")
            continue
        print(f"  {'Emotion':<14} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        print("  " + "-" * 48)
        for cls in class_names:
            d = report.get(cls, {})
            print(
                f"  {cls:<14} "
                f"{d.get('precision', 0)*100:>9.2f}% "
                f"{d.get('recall', 0)*100:>9.2f}% "
                f"{d.get('f1-score', 0)*100:>9.2f}% "
                f"{int(d.get('support', 0)):>10}"
            )


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if not os.path.exists(RESULTS_FILE):
        print(f"\n  No results found at: {RESULTS_FILE}")
        print("  Run train_unified.py first.\n")
        return

    with open(RESULTS_FILE) as f:
        data = json.load(f)

    runs = data.get("runs", [])

    if args.dataset:
        runs = [r for r in runs if r["dataset"] == args.dataset]

    if not runs:
        print("  No matching results found.")
        return

    # Sort
    reverse = SORT_DESC.get(args.sort, True)
    runs = sorted(runs, key=SORT_KEYS[args.sort], reverse=reverse)

    if args.latex:
        print_latex_table(runs)
        return

    print_console_table(runs, args.sort)
    print_per_class_breakdown(runs)
    print_latex_table(runs)


if __name__ == "__main__":
    main()
