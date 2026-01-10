from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


@dataclass
class MAEConfig:
    masking_ratio: float = 0.75
    decoder_depth: int = 8
    decoder_dim: int = 512
    decoder_heads: int = 16
    image_size: int = 224
    patch_size: int = 14
    encoder_dim: int = 1024
    encoder_depth: int = 24
    encoder_heads: int = 16


class MAEPretrainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        mae_config = config.get("mae", {})
        self.mae_cfg = MAEConfig(
            masking_ratio=mae_config.get("masking_ratio", 0.75),
            decoder_depth=mae_config.get("decoder_depth", 8),
            decoder_dim=mae_config.get("decoder_dim", 512),
            decoder_heads=mae_config.get("decoder_heads", 16),
        )

        training_config = config.get("training", {})
        self.batch_size = training_config.get("batch_size", 256)
        self.epochs = training_config.get("epochs", 800)
        self.lr = training_config.get("learning_rate", 1.5e-4)
        self.warmup_epochs = training_config.get("warmup_epochs", 40)
        self.min_lr = training_config.get("min_lr", 1e-6)

        optimizer_config = config.get("optimizer", {})
        self.weight_decay = optimizer_config.get("weight_decay", 0.05)

        checkpoint_config = config.get("checkpoint", {})
        self.save_interval = checkpoint_config.get("save_interval", 50)
        self.output_dir = Path(checkpoint_config.get("output_dir", "pretrained/mae"))

        self.model: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None

    def build_model(self) -> nn.Module:
        return MAEModel(self.mae_cfg)

    def train(self, dataloader: DataLoader) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model().to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0

            lr = self._get_lr(epoch)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            for batch in pbar:
                images = (
                    batch[0].to(device)
                    if isinstance(batch, (list, tuple))
                    else batch.to(device)
                )

                self.optimizer.zero_grad()
                loss = self.model(images)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.2e}"})

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}: avg_loss={avg_loss:.4f}")

            if (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint(epoch + 1)

        self._save_checkpoint(self.epochs, final=True)

    def _get_lr(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return self.lr * epoch / self.warmup_epochs

        progress = (epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)
        return self.min_lr + 0.5 * (self.lr - self.min_lr) * (
            1 + torch.cos(torch.tensor(progress * 3.14159)).item()
        )

    def _save_checkpoint(self, epoch: int, final: bool = False) -> None:
        if self.model is None:
            return

        filename = "final.pth" if final else f"checkpoint_{epoch:04d}.pth"
        path = self.output_dir / filename

        encoder_state = {}
        for name, param in self.model.state_dict().items():
            if name.startswith("encoder"):
                encoder_state[name.replace("encoder.", "")] = param

        torch.save(encoder_state, path)
        print(f"Saved checkpoint: {path}")


class MAEModel(nn.Module):
    def __init__(self, config: MAEConfig) -> None:
        super().__init__()
        self.config = config
        self.num_patches = (config.image_size // config.patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            3, config.encoder_dim, config.patch_size, config.patch_size
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, config.encoder_dim)
        )

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.encoder_dim,
                nhead=config.encoder_heads,
                dim_feedforward=config.encoder_dim * 4,
                batch_first=True,
            ),
            num_layers=config.encoder_depth,
        )

        self.decoder_embed = nn.Linear(config.encoder_dim, config.decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, config.decoder_dim)
        )

        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.decoder_dim,
                nhead=config.decoder_heads,
                dim_feedforward=config.decoder_dim * 4,
                batch_first=True,
            ),
            num_layers=config.decoder_depth,
        )

        self.pred = nn.Linear(config.decoder_dim, config.patch_size**2 * 3)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        target = self._patchify(x)

        patches = self.patch_embed(x).flatten(2).transpose(1, 2)
        patches = patches + self.pos_embed

        num_masked = int(self.num_patches * self.config.masking_ratio)
        noise = torch.rand(batch_size, self.num_patches, device=x.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)

        ids_keep = ids_shuffle[:, : self.num_patches - num_masked]
        visible = torch.gather(
            patches, 1, ids_keep.unsqueeze(-1).expand(-1, -1, patches.shape[-1])
        )

        encoded = self.encoder(visible)

        decoded = self.decoder_embed(encoded)
        mask_tokens = self.mask_token.expand(batch_size, num_masked, -1)
        full_tokens = torch.cat([decoded, mask_tokens], dim=1)

        full_tokens = torch.gather(
            full_tokens,
            1,
            ids_restore.unsqueeze(-1).expand(-1, -1, full_tokens.shape[-1]),
        )
        full_tokens = full_tokens + self.decoder_pos_embed

        decoded = self.decoder(full_tokens)
        pred = self.pred(decoded)

        mask = torch.ones(batch_size, self.num_patches, device=x.device)
        mask[:, : self.num_patches - num_masked] = 0
        mask = torch.gather(mask, 1, ids_restore)

        loss = ((pred - target) ** 2).mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()

        return loss

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        p = self.config.patch_size
        h = w = self.config.image_size // p
        x = x.reshape(x.shape[0], 3, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(x.shape[0], h * w, p * p * 3)
        return x
