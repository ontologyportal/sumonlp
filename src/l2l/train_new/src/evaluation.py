import os
import lightning.pytorch as pl
import hydra
import pyrootutils
import torch
from pathlib import Path
import sys

from omegaconf import OmegaConf

# Calculate the project root (one level up from the src directory)
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.modules.model import L2LModel

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["configs"],
    pythonpath=True,
    dotenv=True,
)



torch.set_float32_matmul_precision(
    "medium"
) # for better performance on multi-GPU training

# Set to make webdataset .lock file readable by other users
os.umask(0)

@hydra.main(version_base="1.3", config_path=str(root / "configs"), config_name="config")
def main(cfg):

    if not os.path.exists(cfg.module.load_model_path):
        raise FileNotFoundError(f"No checkpoint found at {cfg.module.load_model_path}")

    model = L2LModel.load_from_checkpoint(
          cfg.module.load_model_path
    )

    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    fit_kwargs = {
        "model": model,
        "datamodule": datamodule
    }

    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer)

    print("Running test evaluation...")
    # datamodule.setup(stage="test")  # Ensure test dataset is initialized
    results = trainer.test(**fit_kwargs)  # Capture test results
    print("Test Results:", results)


if __name__ == "__main__":
    main()