"""
resnet_bc.py

Simple BC Model with a ResNet-34 backbone for processing inputs; performs late fusion with proprioceptive state
(end-effector), then predicts a actions directly.
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim import AdamW, Optimizer
from torchvision.models import resnet34


class ResNetBC(LightningModule):
    def __init__(self, proprioceptive_dim: int, action_dim: int, hidden_dim: int = 64, resnet: str = "resnet34") -> None:
        super().__init__()
        self.resnet_dim = {"resnet34": 512}[resnet]
        self.obs_dim = self.resnet_dim + proprioceptive_dim

        # Fetch Frozen ResNet --> set resnet34.fc to the Identity function to preserve the embedding!
        print("\t=>> Fetching Pretrained ResNet-34 Backbone...")
        self.resnet34 = resnet34(pretrained=True, progress=True)
        self.resnet34.fc = nn.Identity()
        for param in self.resnet34.parameters():
            param.requires_grad = False

        # Create MLP --> Preceded by a BatchNorm1D
        self.bn = nn.BatchNorm1d(self.obs_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # From R3M Code -- Initialize last layer weights to be small?
        for p in list(self.mlp.parameters())[-2:]:
            p.data *= 1e-2

    def forward(self, imgs: torch.Tensor, proprioceptive_state: torch.Tensor) -> torch.Tensor:
        """Encode images with the ResNet (no gradient), followed by concatenation, batch norm, and projection."""
        with torch.no_grad():
            img_embeddings = self.resnet34(imgs)

        # Concatenate Image Embeddings & Proprioceptive State
        observed = torch.cat([img_embeddings, proprioceptive_state], dim=1)
        normalized = self.bn(observed)

        # Return action from MLP
        return self.mlp(normalized)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Unroll batch, perform a forward pass, and compute MSE Loss with respect to actual actions."""
        imgs, proprioceptive_state, actions = batch
        predicted_actions = self.forward(imgs, proprioceptive_state)

        # Measure MSE Loss
        loss = F.mse_loss(predicted_actions, actions)

        # Log Loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Unroll batch, perform a forward pass, and compute MSE Loss with respect to actual actions."""
        imgs, proprioceptive_state, actions = batch
        predicted_actions = self.forward(imgs, proprioceptive_state)

        # Measure MSE Loss
        loss = F.mse_loss(predicted_actions, actions)

        # Log Loss
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self) -> Optimizer:
        return AdamW([p for p in self.parameters() if p.requires_grad])
