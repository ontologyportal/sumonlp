import os
import torch
import lightning.pytorch as pl
import hydra
import pyrootutils
from pathlib import Path
from transformers import set_seed

# Define project root
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["configs"],
    pythonpath=True,
    dotenv=True,
)

# Set float32 precision for better performance
torch.set_float32_matmul_precision("medium")

# Ensure the .lock file is accessible by other users
os.umask(0)

def train(cfg):

    preprocessor = hydra.utils.instantiate(cfg.preprocessor)
    preprocessor.preprocess_and_save()

    # Initialize DataModule
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    model: pl.LightningModule = hydra.utils.instantiate(cfg.module)

    # Initialize Trainer
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer)

    # Train the model
    trainer.fit(model, datamodule)

    # Test the model after training
    # trainer.test(model, datamodule)

@hydra.main(version_base="1.3", config_path=str(root / "configs"))
def main(cfg):
    pl.seed_everything(42)
    train(cfg)

if __name__ == "__main__":
    main()
