#!/usr/bin/env python3
"""
Analyze experiment results and generate comprehensive report.
"""

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def load_experiment_results(results_dir: Path) -> list[dict]:
    experiments = []
    for exp_dir in results_dir.iterdir():
        if exp_dir.is_dir():
            results_file = exp_dir / "results.json"
            if results_file.exists():
                with open(results_file) as f:
                    experiments.append(json.load(f))
    return experiments


def format_category_table(results: list[dict]) -> str:
    categories = [r["category"] for r in results[0]["per_category_results"]]

    lines = []
    header = "| Category |"
    separator = "|----------|"

    for exp in results:
        name = f"{exp['backbone'].split('_')[0]}+{exp['head'][:4]}"
        header += f" {name:>12} |"
        separator += "-------------:|"

    lines.append(header)
    lines.append(separator)

    for cat in categories:
        row = f"| {cat:<8} |"
        for exp in results:
            cat_result = next(
                (r for r in exp["per_category_results"] if r["category"] == cat), None
            )
            if cat_result:
                auroc = cat_result["image_auroc"] * 100
                row += f" {auroc:>10.2f}% |"
            else:
                row += f" {'N/A':>11} |"
        lines.append(row)

    avg_row = "| **Avg** |"
    for exp in results:
        avg_row += f" **{exp['avg_image_auroc'] * 100:>8.2f}%** |"
    lines.append(avg_row)

    return "\n".join(lines)


def format_summary_table(results: list[dict]) -> str:
    lines = [
        "| Backbone | Head | Avg AUROC | Avg Prec@100R | Best Cat | Worst Cat |",
        "|----------|------|-----------|---------------|----------|-----------|",
    ]

    for exp in sorted(results, key=lambda x: x["avg_image_auroc"], reverse=True):
        per_cat = exp["per_category_results"]
        best = max(per_cat, key=lambda x: x["image_auroc"])
        worst = min(per_cat, key=lambda x: x["image_auroc"])

        lines.append(
            f"| {exp['backbone']} | {exp['head']} | "
            f"{exp['avg_image_auroc'] * 100:.2f}% | "
            f"{exp.get('avg_precision_at_100_recall', 0) * 100:.2f}% | "
            f"{best['category']} ({best['image_auroc'] * 100:.1f}%) | "
            f"{worst['category']} ({worst['image_auroc'] * 100:.1f}%) |"
        )

    return "\n".join(lines)


def generate_report(results_dirs: list[Path], output_file: Path) -> None:
    all_results = []
    for results_dir in results_dirs:
        if results_dir.exists():
            all_results.extend(load_experiment_results(results_dir))

    if not all_results:
        print("No results found!")
        return

    report_lines = [
        "# MVTec AD Experiment Results Report",
        "",
        "## Summary",
        "",
        format_summary_table(all_results),
        "",
        "## Per-Category Results",
        "",
    ]

    backbone_groups = {}
    for exp in all_results:
        backbone = exp["backbone"]
        if backbone not in backbone_groups:
            backbone_groups[backbone] = []
        backbone_groups[backbone].append(exp)

    for backbone, exps in backbone_groups.items():
        report_lines.append(f"### {backbone}")
        report_lines.append("")
        report_lines.append(format_category_table(exps))
        report_lines.append("")

    best_exp = max(all_results, key=lambda x: x["avg_image_auroc"])
    report_lines.extend(
        [
            "## Best Configuration",
            "",
            f"**{best_exp['backbone']} + {best_exp['head']}**",
            f"- Average AUROC: {best_exp['avg_image_auroc'] * 100:.2f}%",
            f"- Average Precision@100%Recall: {best_exp.get('avg_precision_at_100_recall', 0) * 100:.2f}%",
            "",
            "### Per-Category Breakdown",
            "",
            "| Category | AUROC | Precision@100R |",
            "|----------|-------|----------------|",
        ]
    )

    for r in sorted(
        best_exp["per_category_results"], key=lambda x: x["image_auroc"], reverse=True
    ):
        report_lines.append(
            f"| {r['category']} | {r['image_auroc'] * 100:.2f}% | "
            f"{r.get('precision_at_100_recall', 0) * 100:.2f}% |"
        )

    report_lines.extend(
        [
            "",
            "## Conclusions",
            "",
            "1. Best overall configuration for MVTec AD anomaly detection",
            "2. Categories with perfect detection (100% AUROC)",
            "3. Challenging categories requiring further optimization",
            "",
        ]
    )

    with open(output_file, "w") as f:
        f.write("\n".join(report_lines))

    print(f"Report generated: {output_file}")


def main():
    results_dirs = [
        _PROJECT_ROOT / "results" / "dinov3_full",
        _PROJECT_ROOT / "results" / "swin_full",
        _PROJECT_ROOT / "results" / "unified",
    ]

    output_file = _PROJECT_ROOT / "docs" / "EXPERIMENT_RESULTS.md"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    generate_report(results_dirs, output_file)

    print("\n" + "=" * 60)
    with open(output_file) as f:
        print(f.read())


if __name__ == "__main__":
    main()
