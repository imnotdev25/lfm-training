"""
Centralized configuration dataclass for the training pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


def _resolve_hf_token(cli_token: Optional[str] = None) -> Optional[str]:
    """Resolve the Hugging Face token with the following priority:
    1. Explicit CLI argument
    2. HF_TOKEN environment variable
    3. Kaggle Secrets (kaggle_secrets.UserSecretsClient)
    """
    if cli_token:
        return cli_token

    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        return env_token

    # Attempt Kaggle secrets
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore[import-untyped]

        user_secrets = UserSecretsClient()
        kaggle_token = user_secrets.get_secret("HF_TOKEN")
        if kaggle_token:
            return kaggle_token
    except (ImportError, Exception):
        pass

    return None


def _resolve_wandb_key(cli_key: Optional[str] = None) -> Optional[str]:
    """Resolve the Weights & Biases API key with the following priority:
    1. Explicit CLI argument
    2. WANDB_API_KEY environment variable
    3. Kaggle Secrets (kaggle_secrets.UserSecretsClient)
    """
    if cli_key:
        return cli_key

    env_key = os.environ.get("WANDB_API_KEY")
    if env_key:
        return env_key

    # Attempt Kaggle secrets
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore[import-untyped]

        user_secrets = UserSecretsClient()
        kaggle_key = user_secrets.get_secret("WANDB_API_KEY")
        if kaggle_key:
            return kaggle_key
    except (ImportError, Exception):
        pass

    return None


@dataclass
class TrainingConfig:
    """All knobs for a single training run."""

    # ── Model ──────────────────────────────────────────────────────────
    model_name: str = "liquid/LFM2.5-1.2B-Base"
    trust_remote_code: bool = True
    resume_from_model: Optional[str] = None  # Path or Hub ID of a previously trained adapter/model

    # ── Dataset ────────────────────────────────────────────────────────
    dataset_paths: list[str] = field(default_factory=list)
    dataset_text_column: str = "text"
    max_seq_length: int = 2048
    tool_calling_only: bool = False
    quality_filter: bool = False
    eval_split: float = 0.0  # 0.0 = no eval split; 0.1 = 10% held out

    # ── LoRA / PEFT ────────────────────────────────────────────────────
    use_lora: bool = True  # False = full fine-tuning (all parameters)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # ── Training hyper-params ──────────────────────────────────────────
    output_dir: str = "./lfm-checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    fp16: bool = True
    bf16: bool = False
    logging_steps: int = 10
    save_steps: int = 100
    save_strategy: str = "steps"
    save_total_limit: int = 3
    report_to: str = "none"  # "none", "wandb", "tensorboard"
    wandb_api_key: Optional[str] = None
    wandb_project: str = "lfm-trainer"

    # ── Hub / publishing ───────────────────────────────────────────────
    hf_token: Optional[str] = None
    hub_repo_id: Optional[str] = None
    push_to_hub: bool = True

    # ── Post-training export ───────────────────────────────────────────
    export_gguf: bool = False
    export_mlx: bool = False
    export_output_dir: str = "./lfm-exports"

    # ── Benchmarking ──────────────────────────────────────────────────
    run_benchmark: bool = False
    benchmark_before_after: bool = False  # Run on base model too for comparison
    benchmark_max_problems: Optional[int] = None  # Cap problems for quick testing
    benchmark_names: Optional[list[str]] = None  # None = ["humaneval", "mbpp"]; pass ["all"] for all
    generate_model_card: bool = True

    # ── DPO / Preference Alignment ────────────────────────────────────
    dpo_dataset: Optional[str] = None       # HF dataset with prompt/chosen/rejected
    dpo_beta: float = 0.1                   # DPO β — higher = more conservative
    dpo_epochs: int = 1
    dpo_learning_rate: float = 5e-5
    dpo_batch_size: int = 2
    dpo_sft_model: Optional[str] = None     # Override path to SFT adapter for DPO

    # ── Auto Hyperparameter Search ────────────────────────────────────
    auto_hp_search: bool = False
    hp_search_trials_steps: int = 50        # Steps per trial

    # ── Debug ──────────────────────────────────────────────────────────
    simulate_error: bool = False

    def __post_init__(self) -> None:
        self.hf_token = _resolve_hf_token(self.hf_token)

        # Auto-resolve and login W&B if needed
        if self.report_to == "wandb":
            self.wandb_api_key = _resolve_wandb_key(self.wandb_api_key)
            if self.wandb_api_key:
                os.environ["WANDB_API_KEY"] = self.wandb_api_key
                os.environ.setdefault("WANDB_PROJECT", self.wandb_project)
                try:
                    import wandb
                    wandb.login(key=self.wandb_api_key, relogin=True)
                except ImportError:
                    pass
