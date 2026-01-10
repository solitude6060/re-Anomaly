"""
MVTec AD and MVTec LOCO dataset download and preprocessing utilities.

MVTec AD: https://www.mvtec.com/company/research/datasets/mvtec-ad
MVTec LOCO: https://www.mvtec.com/company/research/datasets/mvtec-loco

Note: These datasets require manual download from the official website
due to license requirements. This script provides preprocessing utilities.
"""

import hashlib
import shutil
import tarfile
from pathlib import Path
from typing import Any

MVTEC_AD_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

MVTEC_LOCO_CATEGORIES = [
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors",
]

MVTEC_AD_INFO = {
    "url": "https://www.mvtec.com/company/research/datasets/mvtec-ad",
    "filename": "mvtec_anomaly_detection.tar.xz",
    "size_gb": 4.9,
    "num_categories": 15,
    "total_images": 5354,
}

MVTEC_LOCO_INFO = {
    "url": "https://www.mvtec.com/company/research/datasets/mvtec-loco",
    "filename": "mvtec_loco_anomaly_detection.tar.xz",
    "size_gb": 1.1,
    "num_categories": 5,
    "total_images": 3644,
}


def extract_dataset(archive_path: Path, output_dir: Path) -> None:
    print(f"Extracting {archive_path} to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    if archive_path.suffix == ".xz" or str(archive_path).endswith(".tar.xz"):
        import lzma

        with lzma.open(archive_path) as xz:
            with tarfile.open(fileobj=xz) as tar:
                tar.extractall(output_dir)
    elif archive_path.suffix in (".tar", ".gz", ".tgz"):
        with tarfile.open(archive_path) as tar:
            tar.extractall(output_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

    print(f"Extraction complete: {output_dir}")


def verify_mvtec_ad_structure(data_dir: Path) -> dict[str, Any]:
    results = {"valid": True, "categories": {}, "errors": []}

    for category in MVTEC_AD_CATEGORIES:
        cat_dir = data_dir / category
        if not cat_dir.exists():
            results["valid"] = False
            results["errors"].append(f"Missing category: {category}")
            continue

        train_dir = cat_dir / "train" / "good"
        test_dir = cat_dir / "test"
        gt_dir = cat_dir / "ground_truth"

        cat_info = {
            "train_good": len(list(train_dir.glob("*.png")))
            if train_dir.exists()
            else 0,
            "test_good": len(list((test_dir / "good").glob("*.png")))
            if (test_dir / "good").exists()
            else 0,
            "test_anomaly": 0,
            "has_ground_truth": gt_dir.exists(),
        }

        if test_dir.exists():
            for defect_dir in test_dir.iterdir():
                if defect_dir.is_dir() and defect_dir.name != "good":
                    cat_info["test_anomaly"] += len(list(defect_dir.glob("*.png")))

        results["categories"][category] = cat_info

    return results


def verify_mvtec_loco_structure(data_dir: Path) -> dict[str, Any]:
    results = {"valid": True, "categories": {}, "errors": []}

    for category in MVTEC_LOCO_CATEGORIES:
        cat_dir = data_dir / category
        if not cat_dir.exists():
            results["valid"] = False
            results["errors"].append(f"Missing category: {category}")
            continue

        train_dir = cat_dir / "train" / "good"
        test_dir = cat_dir / "test"
        gt_dir = cat_dir / "ground_truth"

        cat_info = {
            "train_good": len(list(train_dir.glob("*.png")))
            if train_dir.exists()
            else 0,
            "test_good": len(list((test_dir / "good").glob("*.png")))
            if (test_dir / "good").exists()
            else 0,
            "test_logical": 0,
            "test_structural": 0,
            "has_ground_truth": gt_dir.exists(),
        }

        if test_dir.exists():
            for defect_dir in test_dir.iterdir():
                if defect_dir.is_dir() and defect_dir.name != "good":
                    count = len(list(defect_dir.glob("*.png")))
                    if "logical" in defect_dir.name:
                        cat_info["test_logical"] += count
                    else:
                        cat_info["test_structural"] += count

        results["categories"][category] = cat_info

    return results


def print_dataset_info(name: str, results: dict[str, Any]) -> None:
    print(f"\n{'=' * 60}")
    print(f"{name} Dataset Verification")
    print(f"{'=' * 60}")

    if not results["valid"]:
        print(f"Status: INVALID")
        for error in results["errors"]:
            print(f"  - {error}")
        return

    print(f"Status: VALID")
    print(f"\nCategory Statistics:")
    print(f"{'Category':<20} {'Train':>8} {'Test OK':>8} {'Test Anom':>10}")
    print("-" * 50)

    total_train = 0
    total_test_ok = 0
    total_test_anom = 0

    for cat, info in results["categories"].items():
        train = info["train_good"]
        test_ok = info["test_good"]
        test_anom = (
            info.get("test_anomaly", 0)
            + info.get("test_logical", 0)
            + info.get("test_structural", 0)
        )

        total_train += train
        total_test_ok += test_ok
        total_test_anom += test_anom

        print(f"{cat:<20} {train:>8} {test_ok:>8} {test_anom:>10}")

    print("-" * 50)
    print(f"{'TOTAL':<20} {total_train:>8} {total_test_ok:>8} {total_test_anom:>10}")


def create_download_instructions() -> str:
    return """
================================================================================
MVTec Dataset Download Instructions
================================================================================

Due to license requirements, MVTec datasets must be downloaded manually.

1. MVTec AD (Anomaly Detection):
   - Visit: https://www.mvtec.com/company/research/datasets/mvtec-ad
   - Download: mvtec_anomaly_detection.tar.xz (~4.9 GB)
   - Place in: data/mvtec_ad/

2. MVTec LOCO AD (Logical Constraints):
   - Visit: https://www.mvtec.com/company/research/datasets/mvtec-loco
   - Download: mvtec_loco_anomaly_detection.tar.xz (~1.1 GB)
   - Place in: data/mvtec_loco/

After downloading, run:
    python scripts/prepare_datasets.py --extract

================================================================================
"""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MVTec dataset utilities")
    parser.add_argument(
        "--extract", action="store_true", help="Extract downloaded archives"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify dataset structure"
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if args.extract:
        mvtec_ad_archive = data_dir / "mvtec_ad" / "mvtec_anomaly_detection.tar.xz"
        mvtec_loco_archive = (
            data_dir / "mvtec_loco" / "mvtec_loco_anomaly_detection.tar.xz"
        )

        if mvtec_ad_archive.exists():
            extract_dataset(mvtec_ad_archive, data_dir / "mvtec_ad")
        else:
            print(f"MVTec AD archive not found: {mvtec_ad_archive}")

        if mvtec_loco_archive.exists():
            extract_dataset(mvtec_loco_archive, data_dir / "mvtec_loco")
        else:
            print(f"MVTec LOCO archive not found: {mvtec_loco_archive}")

    if args.verify:
        mvtec_ad_dir = data_dir / "mvtec_ad" / "mvtec_anomaly_detection"
        mvtec_loco_dir = data_dir / "mvtec_loco" / "mvtec_loco_anomaly_detection"

        if mvtec_ad_dir.exists():
            results = verify_mvtec_ad_structure(mvtec_ad_dir)
            print_dataset_info("MVTec AD", results)
        else:
            print(f"MVTec AD not found at: {mvtec_ad_dir}")

        if mvtec_loco_dir.exists():
            results = verify_mvtec_loco_structure(mvtec_loco_dir)
            print_dataset_info("MVTec LOCO", results)
        else:
            print(f"MVTec LOCO not found at: {mvtec_loco_dir}")

    if not args.extract and not args.verify:
        print(create_download_instructions())
