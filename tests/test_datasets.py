"""Tests for the Universal Dataset Interface - TDD approach."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from src.data import (
    AnomalyDataset,
    DatasetFactory,
    FewShotSampler,
    MVTecADDataset,
)


@pytest.fixture
def temp_mvtec_structure(tmp_path: Path) -> Path:
    category = "bottle"
    train_good = tmp_path / category / "train" / "good"
    train_good.mkdir(parents=True)
    for i in range(5):
        img = Image.new("RGB", (256, 256), color="white")
        img.save(train_good / f"{i:03d}.png")

    test_good = tmp_path / category / "test" / "good"
    test_good.mkdir(parents=True)
    for i in range(2):
        img = Image.new("RGB", (256, 256), color="white")
        img.save(test_good / f"{i:03d}.png")

    test_defect = tmp_path / category / "test" / "broken_large"
    test_defect.mkdir(parents=True)
    gt_defect = tmp_path / category / "ground_truth" / "broken_large"
    gt_defect.mkdir(parents=True)
    for i in range(3):
        img = Image.new("RGB", (256, 256), color="red")
        img.save(test_defect / f"{i:03d}.png")
        mask = Image.new("L", (256, 256), color="white")
        mask.save(gt_defect / f"{i:03d}_mask.png")

    return tmp_path


@pytest.fixture
def temp_visa_structure(tmp_path: Path) -> Path:
    category = "pcb1"
    train_dir = tmp_path / category / "train" / "good"
    train_dir.mkdir(parents=True)
    for i in range(5):
        img = Image.new("RGB", (512, 512), color="white")
        img.save(train_dir / f"000{i}.JPG")

    test_good = tmp_path / category / "test" / "good"
    test_good.mkdir(parents=True)
    for i in range(2):
        img = Image.new("RGB", (512, 512), color="white")
        img.save(test_good / f"000{i}.JPG")

    test_defect = tmp_path / category / "test" / "anomaly"
    test_defect.mkdir(parents=True)
    gt_dir = tmp_path / category / "ground_truth" / "anomaly"
    gt_dir.mkdir(parents=True)
    for i in range(3):
        img = Image.new("RGB", (512, 512), color="red")
        img.save(test_defect / f"000{i}.JPG")
        mask = Image.new("L", (512, 512), color="white")
        mask.save(gt_dir / f"000{i}.png")

    return tmp_path


class TestAnomalyDatasetProtocol:
    def test_mvtec_ad_implements_protocol(self, temp_mvtec_structure: Path):
        dataset = MVTecADDataset(
            root=temp_mvtec_structure,
            category="bottle",
            split="train",
        )
        assert isinstance(dataset, AnomalyDataset)

    def test_dataset_returns_required_keys(self, temp_mvtec_structure: Path):
        dataset = MVTecADDataset(
            root=temp_mvtec_structure,
            category="bottle",
            split="test",
        )
        sample = dataset[0]

        required_keys = {"image", "label", "label_name", "mask", "image_path"}
        assert required_keys.issubset(sample.keys())

        assert isinstance(sample["image"], torch.Tensor)
        assert isinstance(sample["label"], int)
        assert isinstance(sample["label_name"], str)
        assert isinstance(sample["mask"], torch.Tensor)
        assert isinstance(sample["image_path"], str)

    def test_dataset_image_shape(self, temp_mvtec_structure: Path):
        dataset = MVTecADDataset(
            root=temp_mvtec_structure,
            category="bottle",
            split="train",
            image_size=224,
        )
        sample = dataset[0]
        assert sample["image"].shape == (3, 224, 224)
        assert sample["mask"].shape == (1, 224, 224)


class TestDatasetFactory:
    def test_create_mvtec_ad(self, temp_mvtec_structure: Path):
        dataset = DatasetFactory.create(
            name="mvtec_ad",
            root=temp_mvtec_structure,
            category="bottle",
            split="train",
        )
        assert isinstance(dataset, MVTecADDataset)
        assert len(dataset) == 5

    def test_create_with_custom_image_size(self, temp_mvtec_structure: Path):
        dataset = DatasetFactory.create(
            name="mvtec_ad",
            root=temp_mvtec_structure,
            category="bottle",
            split="train",
            image_size=448,
        )
        sample = dataset[0]
        assert sample["image"].shape == (3, 448, 448)

    def test_invalid_dataset_name_raises(self, temp_mvtec_structure: Path):
        with pytest.raises(ValueError, match="Unknown dataset"):
            DatasetFactory.create(
                name="unknown_dataset",
                root=temp_mvtec_structure,
                category="bottle",
            )

    def test_list_available_datasets(self):
        available = DatasetFactory.list_available()
        assert "mvtec_ad" in available
        assert "mvtec_loco" in available


class TestFewShotSampler:
    def test_sample_k_shots(self, temp_mvtec_structure: Path):
        dataset = MVTecADDataset(
            root=temp_mvtec_structure,
            category="bottle",
            split="train",
        )
        sampler = FewShotSampler(dataset, k=3, seed=42)
        indices = list(sampler)
        assert len(indices) == 3

    def test_sample_reproducible_with_seed(self, temp_mvtec_structure: Path):
        dataset = MVTecADDataset(
            root=temp_mvtec_structure,
            category="bottle",
            split="train",
        )
        sampler1 = FewShotSampler(dataset, k=3, seed=42)
        sampler2 = FewShotSampler(dataset, k=3, seed=42)
        assert list(sampler1) == list(sampler2)

    def test_sample_k_larger_than_dataset(self, temp_mvtec_structure: Path):
        dataset = MVTecADDataset(
            root=temp_mvtec_structure,
            category="bottle",
            split="train",
        )
        sampler = FewShotSampler(dataset, k=10, seed=42)
        indices = list(sampler)
        assert len(indices) == 5

    def test_fewshot_subset_creation(self, temp_mvtec_structure: Path):
        dataset = MVTecADDataset(
            root=temp_mvtec_structure,
            category="bottle",
            split="train",
        )
        sampler = FewShotSampler(dataset, k=2, seed=42)
        subset = sampler.get_subset()

        assert len(subset) == 2
        sample = subset[0]
        assert "image" in sample


class TestCustomDataset:
    def test_create_from_folder_structure(self, tmp_path: Path):
        train_good = tmp_path / "train" / "good"
        train_good.mkdir(parents=True)
        test_good = tmp_path / "test" / "good"
        test_good.mkdir(parents=True)
        test_defect = tmp_path / "test" / "defect"
        test_defect.mkdir(parents=True)

        for i in range(3):
            Image.new("RGB", (100, 100), "white").save(train_good / f"{i}.png")
            Image.new("RGB", (100, 100), "white").save(test_good / f"{i}.png")
            Image.new("RGB", (100, 100), "red").save(test_defect / f"{i}.png")

        dataset = DatasetFactory.create(
            name="custom",
            root=tmp_path,
            split="train",
        )
        assert len(dataset) == 3


class TestDatasetMetadata:
    def test_get_categories_mvtec_ad(self):
        categories = DatasetFactory.get_categories("mvtec_ad")
        assert "bottle" in categories
        assert "screw" in categories
        assert len(categories) == 15

    def test_get_categories_mvtec_loco(self):
        categories = DatasetFactory.get_categories("mvtec_loco")
        assert "breakfast_box" in categories
        assert len(categories) == 5


class TestMultiCategoryLoading:
    def test_load_all_categories(self, temp_mvtec_structure: Path):
        category2 = "cable"
        train_good = temp_mvtec_structure / category2 / "train" / "good"
        train_good.mkdir(parents=True)
        for i in range(3):
            Image.new("RGB", (256, 256), "blue").save(train_good / f"{i:03d}.png")

        datasets = DatasetFactory.create_multi(
            name="mvtec_ad",
            root=temp_mvtec_structure,
            categories=["bottle", "cable"],
            split="train",
        )
        assert len(datasets) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
