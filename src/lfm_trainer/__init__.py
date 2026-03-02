"""LFM Trainer — Fine-tune Liquid LFM 2.5 1.2B for coding on Kaggle multi-GPU."""

__version__ = "0.1.0"

from lfm_trainer.trainer import run_training  # noqa: F401
from lfm_trainer.data import load_datasets  # noqa: F401
from lfm_trainer.export import run_exports  # noqa: F401
