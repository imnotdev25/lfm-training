"""LFM Trainer — Fine-tune Liquid LFM 2.5 1.2B for coding on Kaggle multi-GPU."""

__version__ = "0.13.0"

from lfm_trainer.trainer import run_training  # noqa: F401
from lfm_trainer.data import load_datasets  # noqa: F401
from lfm_trainer.export import run_exports  # noqa: F401
from lfm_trainer.benchmark import run_benchmarks  # noqa: F401
from lfm_trainer.dpo import run_alignment, run_dpo, run_ppo, run_grpo  # noqa: F401
from lfm_trainer.merge import merge_adapters  # noqa: F401
from lfm_trainer.hp_search import auto_hp_search  # noqa: F401
from lfm_trainer.cpt import run_cpt, load_raw_texts  # noqa: F401
from lfm_trainer.distill import run_distillation  # noqa: F401
from lfm_trainer.structured_output import (  # noqa: F401
    create_structured_output_dataset,
    validate_json,
    validate_against_schema,
)
