"""
Example: Export-only — quantize a previously trained model to GGUF and MLX.

Useful when you've already trained a model and want to create quantized
versions without re-training.

Usage:
    python examples/export_only.py
"""

from lfm_trainer.export import run_exports

# Path to your trained model (or any HuggingFace model directory)
MODEL_DIR = "./lfm-checkpoints/merged-for-export"

# All quant variants will share this version tag on the Hub
VERSION_TAG = "v1.0.0"

results = run_exports(
    model_dir=MODEL_DIR,
    repo_id_base="your-username/lfm-code",
    version_tag=VERSION_TAG,
    token=None,  # Auto-detected from env or Kaggle secrets
    output_base="./lfm-exports",
    enable_gguf=True,   # Q4_K_M, Q6_K, Q8_0
    enable_mlx=True,    # 4-bit, 6-bit, 8-bit (Apple Silicon only)
)

print("Published repos:")
for fmt, repos in results.items():
    for repo in repos:
        print(f"  [{fmt}] https://huggingface.co/{repo}")
