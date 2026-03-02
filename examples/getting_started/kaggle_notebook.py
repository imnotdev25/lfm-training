# Kaggle Notebook: LFM 2.5 Coding Fine-tune
#
# Copy this into a Kaggle notebook cell-by-cell.
# GPU: P100 or 2×T4 recommended.

# %% Cell 1: Install
# !pip install lfm-trainer

# %% Cell 2: Basic coding fine-tune
# !lfm-train \
#     --dataset iamtarun/python_code_instructions_18k_alpaca \
#     --hub-repo your-username/lfm-code \
#     --epochs 1 \
#     --batch-size 2 \
#     --save-steps 50 \
#     --export-gguf

# %% Cell 2 (alt): Tool-calling fine-tune
# !lfm-train \
#     --dataset jdaddyalbs/playwright-mcp-toolcalling \
#     --tool-calling-only \
#     --hub-repo your-username/lfm-tools \
#     --epochs 3 \
#     --max-seq-length 4096 \
#     --export-gguf

# %% Cell 2 (alt): Multi-dataset with DataClaw conversations
# !lfm-train \
#     --dataset peteromallet/dataclaw-peteromallet \
#     --dataset sahil2801/CodeAlpaca-20k \
#     --hub-repo your-username/lfm-multi \
#     --epochs 2 \
#     --export-gguf
