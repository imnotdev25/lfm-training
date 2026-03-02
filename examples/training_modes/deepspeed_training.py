"""
DeepSpeed ZeRO — distributed training for larger models.

ZeRO-2: Shards optimizer states + gradients across GPUs.
         Best for LoRA fine-tuning on 2×T4 / 2×A100.

ZeRO-3: Shards optimizer + gradients + model weights.
         Best for full fine-tuning 7B+ models.

CLI:
    # LoRA + ZeRO-2 on 2×T4
    lfm-train --dataset sahil2801/CodeAlpaca-20k \\
        --deepspeed zero2 \\
        --hub-repo your-username/lfm-ds-lora

    # Full fine-tune + ZeRO-3
    lfm-train --dataset sahil2801/CodeAlpaca-20k \\
        --deepspeed zero3 \\
        --full-finetune \\
        --hub-repo your-username/lfm-ds-full

    # Custom DeepSpeed config
    lfm-train --dataset sahil2801/CodeAlpaca-20k \\
        --deepspeed /path/to/my_ds_config.json
"""

from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training


# ═══════════════════════════════════════════════════════════════════════
#  ZeRO-2: LoRA fine-tuning on multi-GPU
# ═══════════════════════════════════════════════════════════════════════
cfg_zero2 = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    dataset_paths=["sahil2801/CodeAlpaca-20k"],

    deepspeed="zero2",          # Built-in ZeRO Stage 2 config

    use_lora=True,
    lora_r=32,
    num_train_epochs=3,
    bf16=True,

    hub_repo_id="your-username/lfm-zero2-lora",
)
# run_training(cfg_zero2)


# ═══════════════════════════════════════════════════════════════════════
#  ZeRO-3: Full fine-tuning on multi-GPU (7B+ models)
# ═══════════════════════════════════════════════════════════════════════
cfg_zero3 = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    dataset_paths=["sahil2801/CodeAlpaca-20k"],

    deepspeed="zero3",          # Built-in ZeRO Stage 3 config

    use_lora=False,             # Full fine-tuning — needs ZeRO-3 for memory
    num_train_epochs=2,
    learning_rate=5e-5,
    bf16=True,

    hub_repo_id="your-username/lfm-zero3-full",
)
# run_training(cfg_zero3)
