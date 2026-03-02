"""
Knowledge Distillation — train a smaller student from a larger teacher.

The student learns the teacher's output distribution (soft labels),
which transfers more knowledge than hard labels:
  - Relative probabilities between answers encode knowledge
  - Wrong answers at 10% vs 1% tells the student something useful

Loss: L = α * KL(student || teacher) + (1-α) * CrossEntropy
  - α = 0.5 → balanced blend (default)
  - Temperature = 2.0 → softens distributions for better transfer

CLI:
    lfm-train --distill-teacher meta-llama/Llama-3.2-7B \
        --dataset sahil2801/CodeAlpaca-20k \
        --distill-temperature 2.0 \
        --distill-alpha 0.5 \
        --hub-repo your-username/lfm-distilled
"""

from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training


# ═══════════════════════════════════════════════════════════════════════
#  Basic: Distill 7B teacher → 1.2B student
# ═══════════════════════════════════════════════════════════════════════
cfg_basic = TrainingConfig(
    # Student model (small, trainable)
    model_name="liquid/LFM2.5-1.2B-Base",

    # Teacher model (large, frozen)
    distill_teacher="meta-llama/Llama-3.2-7B",

    # Training data
    dataset_paths=["sahil2801/CodeAlpaca-20k"],

    # Distillation params
    distill_temperature=2.0,    # Higher = softer distributions
    distill_alpha=0.5,          # 50% KL + 50% CE

    # Training config
    num_train_epochs=3,
    learning_rate=2e-4,
    use_lora=True,
    lora_r=32,

    bf16=True,
    hub_repo_id="your-username/lfm-distilled-coding",
)
# run_training(cfg_basic)


# ═══════════════════════════════════════════════════════════════════════
#  With DeepSpeed ZeRO-2 (for multi-GPU)
# ═══════════════════════════════════════════════════════════════════════
cfg_ds = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    distill_teacher="meta-llama/Llama-3.2-7B",
    dataset_paths=["sahil2801/CodeAlpaca-20k"],

    distill_temperature=3.0,    # Even softer
    distill_alpha=0.7,          # Lean more on teacher knowledge

    # DeepSpeed for multi-GPU
    deepspeed="zero2",

    num_train_epochs=2,
    learning_rate=1e-4,
    use_lora=True,
    lora_r=32,
    bf16=True,
)
# run_training(cfg_ds)


# ═══════════════════════════════════════════════════════════════════════
#  Full fine-tune distillation (no LoRA, more capacity)
# ═══════════════════════════════════════════════════════════════════════
cfg_full = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    distill_teacher="meta-llama/Llama-3.2-7B",
    dataset_paths=[
        "sahil2801/CodeAlpaca-20k",
        "Salesforce/xlam-function-calling-60k",
    ],

    distill_temperature=2.0,
    distill_alpha=0.5,

    # Full fine-tune — all params trainable
    use_lora=False,
    deepspeed="zero3",          # ZeRO-3 needed for full fine-tune on multi-GPU

    num_train_epochs=2,
    learning_rate=5e-5,
    bf16=True,

    # Benchmark the distilled model
    run_benchmark=True,
    benchmark_names=["humaneval", "mbpp", "toolcall"],
    benchmark_before_after=True,

    hub_repo_id="your-username/lfm-distilled-full",
    export_gguf=True,
)
# run_training(cfg_full)
