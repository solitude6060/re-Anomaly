#!/usr/bin/env python
"""
SALAD Training Script for MVTec LOCO with W&B Logging

This script runs the official SALAD (Semantics-Aware Logical Anomaly Detection)
training on MVTec LOCO dataset with Weights & Biases experiment tracking.

Usage:
    # Train single category
    python scripts/run_salad.py --category breakfast_box

    # Train all categories
    python scripts/run_salad.py --all

    # Train with W&B logging
    python scripts/run_salad.py --all --wandb

    # Train with ImageNet penalty (requires ImageNet dataset)
    python scripts/run_salad.py --category breakfast_box --imagenet_path /path/to/imagenet/train
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Categories in MVTec LOCO
LOCO_CATEGORIES = [
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors",
]

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
SALAD_ROOT = Path("/tmp/SALAD")
MVTEC_LOCO_PATH = PROJECT_ROOT / "data" / "mvtec_loco"
MVTEC_LOCO_SEG_PATH = PROJECT_ROOT / "data" / "mvtec_loco_composition_maps"
OUTPUT_DIR = PROJECT_ROOT / "results" / "plan_b_salad"
TEACHER_WEIGHTS = SALAD_ROOT / "models" / "teacher_medium.pth"


def parse_salad_output(output: str) -> dict:
    """Parse SALAD training/testing output for metrics."""
    metrics = {}

    # Parse AUROC from test output
    # Example: "Image AUROC: 0.8790"
    auroc_match = re.search(r"Image AUROC[:\s]+([0-9.]+)", output, re.IGNORECASE)
    if auroc_match:
        metrics["image_auroc"] = float(auroc_match.group(1))

    # Parse logical/structural AUROC if present
    logical_match = re.search(r"Logical AUROC[:\s]+([0-9.]+)", output, re.IGNORECASE)
    if logical_match:
        metrics["logical_auroc"] = float(logical_match.group(1))

    structural_match = re.search(
        r"Structural AUROC[:\s]+([0-9.]+)", output, re.IGNORECASE
    )
    if structural_match:
        metrics["structural_auroc"] = float(structural_match.group(1))

    # Parse loss values
    loss_match = re.search(r"Current loss[:\s]+([0-9.]+)", output)
    if loss_match:
        metrics["loss"] = float(loss_match.group(1))

    return metrics


def run_salad_training(
    category: str,
    imagenet_path: str = "none",
    train_steps: int = 70000,
    seed: int = 42,
    wandb_logger=None,
) -> dict:
    """Run SALAD training for a single category."""

    print(f"\n{'=' * 60}")
    print(f"Training SALAD on category: {category}")
    print(f"{'=' * 60}")

    # Build command
    cmd = [
        sys.executable,  # Use the same Python interpreter
        str(SALAD_ROOT / "train_salad.py"),
        "--category",
        category,
        "--output_dir",
        str(OUTPUT_DIR),
        "--weights",
        str(TEACHER_WEIGHTS),
        "--imagenet_train_path",
        imagenet_path,
        "--mvtec_loco_path",
        str(MVTEC_LOCO_PATH),
        "--mvtec_loco_seg_path",
        str(MVTEC_LOCO_SEG_PATH),
        "--train_steps",
        str(train_steps),
        "--seed",
        str(seed),
    ]

    print(f"Command: {' '.join(cmd)}")
    print()

    # Run training
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SALAD_ROOT)

    result = subprocess.run(
        cmd,
        cwd=str(SALAD_ROOT),
        env=env,
        capture_output=False,  # Show output in real-time
    )

    success = result.returncode == 0

    if not success:
        print(f"ERROR: Training failed for {category}")

    return {"category": category, "success": success}


def run_salad_test(category: str, wandb_logger=None) -> dict:
    """Run SALAD evaluation for a single category."""

    print(f"\n{'=' * 60}")
    print(f"Testing SALAD on category: {category}")
    print(f"{'=' * 60}")

    # Build command
    cmd = [
        sys.executable,
        str(SALAD_ROOT / "test_salad.py"),
        "--category",
        category,
        "--output_dir",
        str(OUTPUT_DIR),
        "--weights",
        str(TEACHER_WEIGHTS),
        "--mvtec_loco_path",
        str(MVTEC_LOCO_PATH),
        "--mvtec_loco_seg_path",
        str(MVTEC_LOCO_SEG_PATH),
    ]

    print(f"Command: {' '.join(cmd)}")
    print()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SALAD_ROOT)

    result = subprocess.run(
        cmd,
        cwd=str(SALAD_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )

    success = result.returncode == 0
    output = result.stdout + result.stderr

    # Parse metrics from output
    metrics = parse_salad_output(output)
    metrics["category"] = category
    metrics["success"] = success

    # Log to W&B
    if wandb_logger and metrics:
        log_data = {}
        if "image_auroc" in metrics:
            log_data[f"{category}/image_auroc"] = metrics["image_auroc"]
        if "logical_auroc" in metrics:
            log_data[f"{category}/logical_auroc"] = metrics["logical_auroc"]
        if "structural_auroc" in metrics:
            log_data[f"{category}/structural_auroc"] = metrics["structural_auroc"]
        if log_data:
            wandb_logger.log(log_data)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run SALAD training on MVTec LOCO")
    parser.add_argument(
        "--category",
        "-c",
        type=str,
        choices=LOCO_CATEGORIES,
        help="Category to train on",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Train on all categories",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only run evaluation (requires pretrained weights)",
    )
    parser.add_argument(
        "--imagenet_path",
        "-i",
        type=str,
        default="none",
        help="Path to ImageNet train set for penalty loss (default: none)",
    )
    parser.add_argument(
        "--train_steps",
        "-t",
        type=int,
        default=70000,
        help="Number of training steps (default: 70000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    # W&B arguments
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="re-anomaly",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="W&B run name",
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        nargs="+",
        default=None,
        help="W&B tags",
    )

    args = parser.parse_args()

    # Validate paths
    if not MVTEC_LOCO_PATH.exists():
        print(f"ERROR: MVTec LOCO dataset not found at {MVTEC_LOCO_PATH}")
        sys.exit(1)

    if not MVTEC_LOCO_SEG_PATH.exists():
        print(f"ERROR: Composition maps not found at {MVTEC_LOCO_SEG_PATH}")
        print(
            "Run: cd data && gdown 'https://docs.google.com/uc?id=1yXdOhHLO47wl7pGYjUudvSRsCz84E-Gc' && unzip mvtec_loco_composition_maps.zip"
        )
        sys.exit(1)

    if not TEACHER_WEIGHTS.exists():
        print(f"ERROR: Teacher weights not found at {TEACHER_WEIGHTS}")
        print("Run: cd /tmp/SALAD && ./download_pretrained_models.sh")
        sys.exit(1)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine categories to process
    if args.all:
        categories = LOCO_CATEGORIES
    elif args.category:
        categories = [args.category]
    else:
        print("ERROR: Must specify --category or --all")
        parser.print_help()
        sys.exit(1)

    # Initialize W&B logger
    wandb_logger = None
    if args.wandb:
        try:
            from src.utils.wandb_logger import WandbLogger, WandbConfig

            job_type = "eval" if args.test_only else "train"
            wandb_config = WandbConfig(
                project=args.wandb_project,
                name=args.wandb_name or f"plan_b_salad_{job_type}",
                tags=args.wandb_tags or ["plan_b", "salad", "mvtec_loco", job_type],
                group="plan_b",
                job_type=job_type,
            )
            wandb_logger = WandbLogger(config=wandb_config, enabled=True)

            # Log experiment config
            wandb_logger.log_config(
                {
                    "experiment": "Plan B - SALAD",
                    "method": "salad",
                    "dataset": "mvtec_loco",
                    "categories": categories,
                    "train_steps": args.train_steps,
                    "seed": args.seed,
                    "imagenet_penalty": args.imagenet_path != "none",
                }
            )
        except ImportError:
            print("WARNING: wandb_logger not available, continuing without W&B")

    # Run training/testing
    results = []
    for idx, category in enumerate(categories):
        if args.test_only:
            result = run_salad_test(category, wandb_logger)
        else:
            result = run_salad_training(
                category=category,
                imagenet_path=args.imagenet_path,
                train_steps=args.train_steps,
                seed=args.seed,
                wandb_logger=wandb_logger,
            )
            # Run test after training
            if result["success"]:
                test_result = run_salad_test(category, wandb_logger)
                result.update(test_result)

        results.append(result)

        # Log step to W&B
        if wandb_logger:
            status = 1 if result.get("success", False) else 0
            wandb_logger.log({f"{category}/completed": status}, step=idx)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    successful_results = [r for r in results if r.get("success", False)]

    for result in results:
        status = "✓ SUCCESS" if result.get("success", False) else "✗ FAILED"
        auroc_str = ""
        if "image_auroc" in result:
            auroc_str = f" (AUROC: {result['image_auroc']:.4f})"
        print(f"  {result['category']}: {status}{auroc_str}")

    # Compute and log averages
    if successful_results:
        aurocs = [r["image_auroc"] for r in successful_results if "image_auroc" in r]
        if aurocs:
            avg_auroc = sum(aurocs) / len(aurocs)
            print(f"\n  Average Image AUROC: {avg_auroc:.4f}")

            if wandb_logger:
                wandb_logger.log_summary(
                    {
                        "avg_image_auroc": avg_auroc,
                        "total_categories": len(categories),
                        "successful_categories": len(successful_results),
                    }
                )

    # Save results to JSON
    results_path = OUTPUT_DIR / "wandb_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "experiment": "Plan B - SALAD",
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {results_path}")

    # Finish W&B run
    if wandb_logger:
        wandb_logger.finish()
        print("W&B logging complete.")

    # Exit with error if any failed
    if not all(r.get("success", False) for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
