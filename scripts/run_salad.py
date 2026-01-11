#!/usr/bin/env python
"""
SALAD Training Script for MVTec LOCO

This script runs the official SALAD (Semantics-Aware Logical Anomaly Detection)
training on MVTec LOCO dataset.

Usage:
    # Train single category
    python scripts/run_salad.py --category breakfast_box

    # Train all categories
    python scripts/run_salad.py --all

    # Train with ImageNet penalty (requires ImageNet dataset)
    python scripts/run_salad.py --category breakfast_box --imagenet_path /path/to/imagenet/train
"""

import argparse
import os
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


def run_salad_training(
    category: str,
    imagenet_path: str = "none",
    train_steps: int = 70000,
    seed: int = 42,
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

    if result.returncode != 0:
        print(f"ERROR: Training failed for {category}")
        return {"category": category, "success": False}

    return {"category": category, "success": True}


def run_salad_test(category: str) -> dict:
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
        capture_output=False,
    )

    return {"category": category, "success": result.returncode == 0}


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

    # Run training/testing
    results = []
    for category in categories:
        if args.test_only:
            result = run_salad_test(category)
        else:
            result = run_salad_training(
                category=category,
                imagenet_path=args.imagenet_path,
                train_steps=args.train_steps,
                seed=args.seed,
            )
        results.append(result)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for result in results:
        status = "✓ SUCCESS" if result["success"] else "✗ FAILED"
        print(f"  {result['category']}: {status}")

    # Exit with error if any failed
    if not all(r["success"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
