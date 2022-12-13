"""
train.py

Core training script -- loads and preprocesses, instantiates a Lightning Module, and runs training. For now, just handles
training a simple ResNet-backed policy for a single task via behavioral cloning... notably supports Gripper handling!

Run with: `python train.py`
"""
import os
from pathlib import Path

import shutil
import torch
from core.models import ResNetBC
from core.preprocessing import get_dataset
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from tap import Tap
from torch.utils.data import DataLoader


class ArgumentParser(Tap):
    # fmt: off
    arch: str = "resnet-bc"                                     # Model architecture to train in < resnet-bc | joint-bc >
    state: str = "rgb"                                          # State-Space either RGB (ResNet34) or joint states
    action: str = "joint-delta"                                 # Action-Space in < endeff-delta | ... >

    # Data Parameters
    data: Path = "demos-user-1-joy-naive"                       # Path to demonstrations directory
    n_train: int = 20                                           # Number of demonstrations to train on...
    n_val: int = 4                                              # Number of validation demonstrations (should be 4)

    # Optimization Parameters
    bsz: int = 512                                              # Batch Size for training & validation
    n_epochs: int = 20                                          # Number of training epochs to run

    # Reproducibility
    seed: int = 7                                               # Random seed, for reproducible training
    # fmt: on


def train() -> None:
    # Parse Arguments...
    print("[*] Sharpening DAggers :: Launching =>>>")
    args = ArgumentParser().parse_args()
    print("\t=>> Thunder is good, thunder is impressive; but it is Lightning that does all the work (Mark Twain)...")

    # Set run identifier - based on file path
    run_id = f"{'-'.join(str(args.data).split('-')[-2:])}+{args.arch}-{args.action}-n={args.n_train}-x{args.seed}"
    run_dir = f"runs/{run_id}"
    os.makedirs(run_dir, exist_ok=True)

    # Randomness
    seed_everything(args.seed, workers=True)

    # Create Dataset
    print(f"[*] Creating `{args.state}-{args.action}` dataset...")
    train_dataset, val_dataset = get_dataset(args.state, args.action, args.data, args.n_train, args.n_val,)

    # Create Model
    print(f"[*] Initializing Model `{args.arch}` with state-actions: `{args.state}-{args.action}`...")
    if args.arch == "resnet-bc":
        model = ResNetBC(train_dataset.proprioceptive_dim, train_dataset.action_dim)
    else:
        raise NotImplementedError(f"Model `{args.arch}` not yet implemented!")

    # Create DataLoaders
    print("[*] Creating DataLoaders, Callbacks, and Loggers...")
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.bsz, num_workers=4, shuffle=False)

    # Create Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"runs/{run_id}",
        filename=f"{run_id}+" + "{epoch:02d}-{train_loss:.4f}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    print("[*] Training...")
    trainer = Trainer(
        max_epochs=args.n_epochs,
        gpus=1 if torch.cuda.is_available() else 0,
        logger=None,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_loader, val_loader)

    # Best Model Handling
    shutil.copy(checkpoint_callback.best_model_path, Path(checkpoint_callback.best_model_path).parent / 'best.ckpt')


if __name__ == "__main__":
    train()
