from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class DINOConfig:
    teacher_momentum: float = 0.996
    student_temp: float = 0.1
    teacher_temp: float = 0.04
    warmup_teacher_temp_epochs: int = 30
    center_momentum: float = 0.9
    out_dim: int = 65536
    hidden_dim: int = 2048


class DINODAPTTrainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        dino_config = config.get("dino", {})
        self.dino_cfg = DINOConfig(
            teacher_momentum=dino_config.get("teacher_momentum", 0.996),
            student_temp=dino_config.get("student_temp", 0.1),
            teacher_temp=dino_config.get("teacher_temp", 0.04),
            warmup_teacher_temp_epochs=dino_config.get(
                "warmup_teacher_temp_epochs", 30
            ),
            center_momentum=dino_config.get("center_momentum", 0.9),
        )

        training_config = config.get("training", {})
        self.batch_size = training_config.get("batch_size", 128)
        self.epochs = training_config.get("epochs", 100)
        self.lr = training_config.get("learning_rate", 5e-4)
        self.warmup_epochs = training_config.get("warmup_epochs", 10)
        self.min_lr = training_config.get("min_lr", 1e-6)

        optimizer_config = config.get("optimizer", {})
        self.weight_decay = optimizer_config.get("weight_decay", 0.04)

        checkpoint_config = config.get("checkpoint", {})
        self.save_interval = checkpoint_config.get("save_interval", 10)
        self.output_dir = Path(
            checkpoint_config.get("output_dir", "pretrained/dino_dapt")
        )

        self.student: nn.Module | None = None
        self.teacher: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.center: torch.Tensor | None = None

    def build_models(self, backbone: nn.Module) -> tuple[nn.Module, nn.Module]:
        student = DINOWrapper(backbone, self.dino_cfg)
        teacher = DINOWrapper(backbone, self.dino_cfg)

        for param in teacher.parameters():
            param.requires_grad = False

        teacher.load_state_dict(student.state_dict())
        return student, teacher

    def train(self, dataloader: DataLoader, backbone: nn.Module) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.student, self.teacher = self.build_models(backbone)
        self.student = self.student.to(device)
        self.teacher = self.teacher.to(device)

        self.center = torch.zeros(1, self.dino_cfg.out_dim, device=device)

        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        for epoch in range(self.epochs):
            self.student.train()
            total_loss = 0.0

            lr = self._get_lr(epoch)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            teacher_temp = self._get_teacher_temp(epoch)

            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            for batch in pbar:
                images = batch[0] if isinstance(batch, (list, tuple)) else batch

                if isinstance(images, list):
                    global_views = [v.to(device) for v in images[:2]]
                    local_views = [v.to(device) for v in images[2:]]
                else:
                    global_views = [images.to(device)]
                    local_views = []

                self.optimizer.zero_grad()
                loss = self._compute_loss(global_views, local_views, teacher_temp)
                loss.backward()
                self.optimizer.step()

                self._update_teacher(epoch)
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}: avg_loss={avg_loss:.4f}")

            if (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint(epoch + 1)

        self._save_checkpoint(self.epochs, final=True)

    def _compute_loss(
        self,
        global_views: list[torch.Tensor],
        local_views: list[torch.Tensor],
        teacher_temp: float,
    ) -> torch.Tensor:
        if self.student is None or self.teacher is None or self.center is None:
            raise RuntimeError("Models not initialized")

        all_views = global_views + local_views
        student_out = [self.student(v) for v in all_views]

        with torch.no_grad():
            teacher_out = [self.teacher(v) for v in global_views]
            teacher_out = [(t - self.center) / teacher_temp for t in teacher_out]
            teacher_probs = [F.softmax(t, dim=-1) for t in teacher_out]

            batch_center = torch.cat(
                [t.mean(dim=0, keepdim=True) for t in teacher_out]
            ).mean(dim=0, keepdim=True)
            self.center = (
                self.dino_cfg.center_momentum * self.center
                + (1 - self.dino_cfg.center_momentum) * batch_center
            )

        total_loss = torch.tensor(0.0, device=global_views[0].device)
        n_loss_terms = 0

        for t_idx, t_prob in enumerate(teacher_probs):
            for s_idx, s_out in enumerate(student_out):
                if t_idx == s_idx and s_idx < len(global_views):
                    continue

                s_log_prob = F.log_softmax(s_out / self.dino_cfg.student_temp, dim=-1)
                loss = -torch.sum(t_prob * s_log_prob, dim=-1).mean()
                total_loss = total_loss + loss
                n_loss_terms += 1

        return total_loss / n_loss_terms if n_loss_terms > 0 else total_loss

    def _update_teacher(self, epoch: int) -> None:
        if self.student is None or self.teacher is None:
            return

        m = self.dino_cfg.teacher_momentum
        with torch.no_grad():
            for param_s, param_t in zip(
                self.student.parameters(), self.teacher.parameters()
            ):
                param_t.data = m * param_t.data + (1 - m) * param_s.data

    def _get_lr(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return self.lr * epoch / self.warmup_epochs
        progress = (epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)
        return self.min_lr + 0.5 * (self.lr - self.min_lr) * (
            1 + torch.cos(torch.tensor(progress * 3.14159)).item()
        )

    def _get_teacher_temp(self, epoch: int) -> float:
        if epoch < self.dino_cfg.warmup_teacher_temp_epochs:
            return (
                0.04
                + (self.dino_cfg.teacher_temp - 0.04)
                * epoch
                / self.dino_cfg.warmup_teacher_temp_epochs
            )
        return self.dino_cfg.teacher_temp

    def _save_checkpoint(self, epoch: int, final: bool = False) -> None:
        if self.student is None:
            return

        filename = "final.pth" if final else f"checkpoint_{epoch:04d}.pth"
        path = self.output_dir / filename

        backbone_state = {}
        for name, param in self.student.state_dict().items():
            if name.startswith("backbone."):
                backbone_state[name.replace("backbone.", "")] = param

        torch.save(backbone_state, path)
        print(f"Saved checkpoint: {path}")


class DINOWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, config: DINOConfig) -> None:
        super().__init__()
        self.backbone = backbone

        embed_dim = 768
        if hasattr(backbone, "embed_dim"):
            embed_dim = backbone.embed_dim

        self.head = nn.Sequential(
            nn.Linear(embed_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.out_dim),
        )
        self.head[-1].weight.data.normal_(mean=0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)

        if isinstance(features, list):
            feat = features[-1]
        else:
            feat = features

        if feat.dim() == 4:
            feat = feat.mean(dim=(2, 3))
        elif feat.dim() == 3:
            feat = feat[:, 0]

        return self.head(feat)
