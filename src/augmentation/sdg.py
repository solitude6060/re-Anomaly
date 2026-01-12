from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SAM3AutoSegmentation(nn.Module):
    """
    SAM3 (Segment Anything Model 3) integration for automatic segmentation.
    Generates pseudo-segmentation masks without human labels.
    Used to enhance anomaly detection by providing spatial priors.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.model_type = config.get("model_type", "sam2.1_hiera_large")
        self.points_per_side = config.get("points_per_side", 32)
        self.pred_iou_thresh = config.get("pred_iou_thresh", 0.88)
        self.stability_score_thresh = config.get("stability_score_thresh", 0.95)
        self.min_mask_region_area = config.get("min_mask_region_area", 100)

        self.sam_model = None
        self.mask_generator = None
        self._initialized = False

    def _lazy_init(self) -> None:
        if self._initialized:
            return

        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

            checkpoint_map = {
                "sam2.1_hiera_large": "sam2.1_hiera_large.pt",
                "sam2.1_hiera_base_plus": "sam2.1_hiera_base_plus.pt",
                "sam2.1_hiera_small": "sam2.1_hiera_small.pt",
                "sam2.1_hiera_tiny": "sam2.1_hiera_tiny.pt",
            }

            checkpoint = checkpoint_map.get(self.model_type, "sam2.1_hiera_large.pt")
            self.sam_model = build_sam2(self.model_type, checkpoint)
            self.mask_generator = SAM2AutomaticMaskGenerator(
                model=self.sam_model,
                points_per_side=self.points_per_side,
                pred_iou_thresh=self.pred_iou_thresh,
                stability_score_thresh=self.stability_score_thresh,
                min_mask_region_area=self.min_mask_region_area,
            )
            self._initialized = True
        except ImportError:
            self._use_fallback_segmentation()

    def _use_fallback_segmentation(self) -> None:
        self._initialized = True
        self.mask_generator = None

    def generate_masks(self, image: torch.Tensor) -> list[dict]:
        self._lazy_init()

        if self.mask_generator is None:
            return self._fallback_generate_masks(image)

        if image.dim() == 4:
            image = image[0]

        np_image = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        masks = self.mask_generator.generate(np_image)
        return masks

    def _fallback_generate_masks(self, image: torch.Tensor) -> list[dict]:
        if image.dim() == 4:
            image = image[0]

        h, w = image.shape[1], image.shape[2]
        gray = image.mean(dim=0)

        threshold = gray.median()
        mask = (gray > threshold).float()

        return [
            {
                "segmentation": mask.cpu().numpy(),
                "area": mask.sum().item(),
                "predicted_iou": 0.9,
            }
        ]

    def get_object_mask(self, image: torch.Tensor) -> torch.Tensor:
        masks = self.generate_masks(image)

        if not masks:
            h, w = image.shape[-2], image.shape[-1]
            return torch.ones(h, w, device=image.device)

        largest_mask = max(masks, key=lambda x: x["area"])
        mask_tensor = torch.from_numpy(largest_mask["segmentation"]).float()

        return mask_tensor.to(image.device)

    def get_attention_prior(self, image: torch.Tensor) -> torch.Tensor:
        masks = self.generate_masks(image)

        if not masks:
            h, w = image.shape[-2], image.shape[-1]
            return torch.ones(h, w, device=image.device) * 0.5

        attention = torch.zeros(image.shape[-2], image.shape[-1])

        for mask_data in masks:
            mask = torch.from_numpy(mask_data["segmentation"]).float()
            iou = mask_data.get("predicted_iou", 0.5)
            attention = torch.maximum(attention, mask * iou)

        return attention.to(image.device)


class SyntheticDefectGenerator(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.defect_types = config.get(
            "defect_types",
            ["cutout", "noise", "blur", "color_shift", "texture", "scratch", "stain"],
        )
        self.defect_size_range = config.get("defect_size_range", (0.05, 0.2))
        self.num_defects_range = config.get("num_defects_range", (1, 3))
        self.perlin_noise_scale = config.get("perlin_noise_scale", 6)
        self.scratch_width_range = config.get("scratch_width_range", (1, 5))
        self.scratch_num_points = config.get("scratch_num_points", 10)

    def generate_defect_mask(
        self, h: int, w: int, device: torch.device
    ) -> torch.Tensor:
        mask = torch.zeros(h, w, device=device)

        num_defects = torch.randint(
            self.num_defects_range[0], self.num_defects_range[1] + 1, (1,)
        ).item()

        for _ in range(num_defects):
            size_ratio = (
                torch.rand(1).item()
                * (self.defect_size_range[1] - self.defect_size_range[0])
                + self.defect_size_range[0]
            )

            defect_h = int(h * size_ratio)
            defect_w = int(w * size_ratio)

            top = torch.randint(0, max(1, h - defect_h), (1,)).item()
            left = torch.randint(0, max(1, w - defect_w), (1,)).item()

            shape_type = torch.randint(0, 3, (1,)).item()

            if shape_type == 0:
                mask[top : top + defect_h, left : left + defect_w] = 1.0
            elif shape_type == 1:
                y_coords, x_coords = torch.meshgrid(
                    torch.arange(h, device=device),
                    torch.arange(w, device=device),
                    indexing="ij",
                )
                center_y, center_x = top + defect_h // 2, left + defect_w // 2
                radius = min(defect_h, defect_w) // 2
                distances = (
                    ((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)
                    .float()
                    .sqrt()
                )
                mask = torch.maximum(mask, (distances <= radius).float())
            else:
                perlin = self._generate_perlin_noise(h, w, device)
                threshold = perlin.quantile(0.7)
                region_mask = (perlin > threshold).float()
                mask = torch.maximum(mask, region_mask)

        return mask

    def _generate_perlin_noise(
        self, h: int, w: int, device: torch.device
    ) -> torch.Tensor:
        scale = self.perlin_noise_scale
        noise = torch.randn(h // scale + 2, w // scale + 2, device=device)
        noise = F.interpolate(
            noise.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        return (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

    def apply_defect(
        self, image: torch.Tensor, mask: torch.Tensor, defect_type: str
    ) -> torch.Tensor:
        if defect_type == "cutout":
            fill_value = torch.rand(3, 1, 1, device=image.device) * 0.3
            result = image * (1 - mask.unsqueeze(0)) + fill_value * mask.unsqueeze(0)
        elif defect_type == "noise":
            noise = torch.randn_like(image) * 0.3
            result = image + noise * mask.unsqueeze(0)
            result = result.clamp(0, 1)
        elif defect_type == "blur":
            blurred = F.avg_pool2d(
                image.unsqueeze(0), kernel_size=15, stride=1, padding=7
            ).squeeze(0)
            result = image * (1 - mask.unsqueeze(0)) + blurred * mask.unsqueeze(0)
        elif defect_type == "color_shift":
            shift = (torch.rand(3, 1, 1, device=image.device) - 0.5) * 0.5
            shifted = (image + shift).clamp(0, 1)
            result = image * (1 - mask.unsqueeze(0)) + shifted * mask.unsqueeze(0)
        elif defect_type == "texture":
            texture = self._generate_perlin_noise(
                image.shape[1], image.shape[2], image.device
            )
            texture = texture.unsqueeze(0).expand(3, -1, -1) * 0.3
            result = image + texture * mask.unsqueeze(0)
            result = result.clamp(0, 1)
        elif defect_type == "scratch":
            result = self._apply_scratch(image, mask)
        elif defect_type == "stain":
            result = self._apply_stain(image, mask)
        else:
            result = image

        return result

    def _apply_scratch(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h, w = image.shape[1], image.shape[2]
        device = image.device

        scratch_mask = torch.zeros(h, w, device=device)

        num_scratches = torch.randint(1, 4, (1,)).item()
        for _ in range(num_scratches):
            y_start = torch.randint(0, h, (1,)).item()
            x_start = torch.randint(0, w, (1,)).item()

            points_y = [y_start]
            points_x = [x_start]

            for _ in range(self.scratch_num_points):
                dy = torch.randint(-15, 16, (1,)).item()
                dx = torch.randint(-15, 16, (1,)).item()
                new_y = max(0, min(h - 1, points_y[-1] + dy))
                new_x = max(0, min(w - 1, points_x[-1] + dx))
                points_y.append(new_y)
                points_x.append(new_x)

            for i in range(len(points_y) - 1):
                y0, x0 = points_y[i], points_x[i]
                y1, x1 = points_y[i + 1], points_x[i + 1]

                steps = max(abs(y1 - y0), abs(x1 - x0)) + 1
                for t in range(steps):
                    ratio = t / max(1, steps - 1)
                    y = int(y0 + ratio * (y1 - y0))
                    x = int(x0 + ratio * (x1 - x0))

                    scratch_width = torch.randint(
                        self.scratch_width_range[0],
                        self.scratch_width_range[1] + 1,
                        (1,),
                    ).item()

                    for dy in range(-scratch_width, scratch_width + 1):
                        for dx in range(-scratch_width, scratch_width + 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                scratch_mask[ny, nx] = 1.0

        combined_mask = torch.maximum(mask, scratch_mask)

        scratch_color = torch.rand(3, 1, 1, device=device) * 0.3 + 0.1
        result = image * (
            1 - combined_mask.unsqueeze(0)
        ) + scratch_color * combined_mask.unsqueeze(0)
        return result.clamp(0, 1)

    def _apply_stain(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h, w = image.shape[1], image.shape[2]
        device = image.device

        stain_color = torch.rand(3, device=device) * 0.4 + 0.1

        intensity = self._generate_perlin_noise(h, w, device)
        intensity = intensity * mask

        intensity_3d = intensity.unsqueeze(0).expand(3, -1, -1)
        stain = stain_color.view(3, 1, 1) * intensity_3d

        result = image * (1 - intensity_3d * 0.7) + stain * 0.7
        return result.clamp(0, 1)

    def generate_anomalous_sample(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image.dim() == 4:
            image = image[0]

        h, w = image.shape[1], image.shape[2]
        device = image.device

        mask = self.generate_defect_mask(h, w, device)

        defect_type = self.defect_types[
            torch.randint(0, len(self.defect_types), (1,)).item()
        ]
        anomalous_image = self.apply_defect(image, mask, defect_type)

        return anomalous_image, mask

    def generate_batch(
        self, images: torch.Tensor, anomaly_ratio: float = 0.5
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = images.shape[0]
        num_anomalies = int(batch_size * anomaly_ratio)

        result_images = images.clone()
        masks = torch.zeros(
            batch_size, images.shape[2], images.shape[3], device=images.device
        )
        labels = torch.zeros(batch_size, device=images.device)

        anomaly_indices = torch.randperm(batch_size)[:num_anomalies]

        for idx in anomaly_indices:
            anomalous, mask = self.generate_anomalous_sample(images[idx])
            result_images[idx] = anomalous
            masks[idx] = mask
            labels[idx] = 1.0

        return result_images, masks, labels


class SDGAugmentedTrainer:
    def __init__(
        self,
        sdg: SyntheticDefectGenerator,
        backbone: nn.Module,
        head: nn.Module,
        device: torch.device,
        anomaly_ratio: float = 0.5,
    ) -> None:
        self.sdg = sdg
        self.backbone = backbone
        self.head = head
        self.device = device
        self.anomaly_ratio = anomaly_ratio

    def train_step(
        self,
        images: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        images = images.to(self.device)

        aug_images, masks, labels = self.sdg.generate_batch(
            images, anomaly_ratio=self.anomaly_ratio
        )

        with torch.no_grad():
            normal_features = self.backbone(images)
            aug_features = self.backbone(aug_images)

        contrastive_loss = self._compute_contrastive_loss(
            normal_features, aug_features, labels
        )

        reconstruction_loss = self._compute_reconstruction_loss(
            aug_features, masks, labels
        )

        total_loss = contrastive_loss + 0.5 * reconstruction_loss

        return {
            "total_loss": total_loss,
            "contrastive_loss": contrastive_loss,
            "reconstruction_loss": reconstruction_loss,
        }

    def _compute_contrastive_loss(
        self,
        normal_features: list[torch.Tensor],
        aug_features: list[torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        normal_concat = torch.cat(
            [f.flatten(start_dim=2).mean(dim=2) for f in normal_features], dim=1
        )
        aug_concat = torch.cat(
            [f.flatten(start_dim=2).mean(dim=2) for f in aug_features], dim=1
        )

        normal_concat = F.normalize(normal_concat, p=2, dim=1)
        aug_concat = F.normalize(aug_concat, p=2, dim=1)

        similarity = (normal_concat * aug_concat).sum(dim=1)

        normal_mask = labels == 0
        anomaly_mask = labels == 1

        loss = torch.tensor(0.0, device=self.device)
        if normal_mask.any():
            loss = loss + (1 - similarity[normal_mask]).mean()
        if anomaly_mask.any():
            loss = loss + F.relu(similarity[anomaly_mask] - 0.5).mean()

        return loss

    def _compute_reconstruction_loss(
        self,
        aug_features: list[torch.Tensor],
        masks: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if not hasattr(self.head, "forward") or labels.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        try:
            output = self.head(aug_features)
            anomaly_map = output.get("anomaly_map")

            if anomaly_map is None:
                return torch.tensor(0.0, device=self.device)

            target_size = anomaly_map.shape[-2:]
            masks_resized = F.interpolate(
                masks.unsqueeze(1),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

            anomaly_mask = labels == 1
            if not anomaly_mask.any():
                return torch.tensor(0.0, device=self.device)

            pred = anomaly_map[anomaly_mask]
            target = masks_resized[anomaly_mask]

            loss = F.binary_cross_entropy_with_logits(pred, target, reduction="mean")
            return loss

        except Exception:
            return torch.tensor(0.0, device=self.device)
