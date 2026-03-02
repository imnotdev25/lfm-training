"""
Example: Merge multiple LoRA adapters into a single model.

Use cases:
  - Combine adapters from different training rounds
  - Mix coding + tool-calling + domain adapters
  - Create a "best of all worlds" model before export

Each adapter is merged sequentially: adapter 1 first, then 2, then 3.
You can also apply per-adapter weights to control influence.

CLI usage:
    # Merge 3 adapters from Hub
    lfm-train \
        --merge-adapters user/lfm-code-v1 user/lfm-tools-v2 user/lfm-domain-v3 \
        --merge-output ./lfm-merged \
        --hub-repo user/lfm-merged-final

    # Merge local adapters
    lfm-train \
        --merge-adapters ./round1/final-adapter ./round2/final-adapter \
        --merge-output ./lfm-merged
"""

from lfm_trainer.merge import merge_adapters

# ── Basic: Merge 3 adapters equally ──────────────────────────────────
merge_adapters(
    base_model_name="liquid/LFM2.5-1.2B-Base",
    adapter_paths=[
        "your-username/lfm-code-v1",        # Round 1: coding
        "your-username/lfm-tools-v2",       # Round 2: tool-calling
        "your-username/lfm-domain-v3",      # Round 3: domain-specific
    ],
    output_dir="./lfm-merged",
    push_to_hub=True,
    hub_repo_id="your-username/lfm-merged-final",
)


# ── Advanced: Weighted merge ─────────────────────────────────────────
# merge_adapters(
#     base_model_name="liquid/LFM2.5-1.2B-Base",
#     adapter_paths=[
#         "./round1/final-adapter",  # Coding (most important)
#         "./round2/final-adapter",  # Tool-calling
#         "./round3/final-adapter",  # Domain (least important)
#     ],
#     weights=[1.0, 0.7, 0.3],     # ← scale each adapter's influence
#     output_dir="./lfm-weighted-merge",
# )
