"""
Example: High-efficiency inference using AirLLM on MLX (Apple Silicon).

AirLLM allows running large models on memory-constrained devices by loading 
model layers one-by-one. The MLX version is optimized for Apple Silicon.

Usage:
    python examples/inference/airllm_mlx_inference.py --model_path "your-username/lfm-model-AirLLM-MLX"
"""

import argparse
import sys

try:
    from airllm import AirLLMMLX
except ImportError:
    print("Error: 'airllm' is not installed or 'AirLLMMLX' is not available.")
    print("Install with: pip install airllm mlx")
    sys.exit(1)

def run_inference(model_path: str, prompt: str):
    print(f"Loading AirLLM-MLX model from: {model_path}")
    
    # Initialize the AirLLM MLX model
    # Note: If it's a Hub ID, it will download and shard it if not already done.
    # If it's a local path to a sharded model (from our export), it will load it directly.
    model = AirLLMMLX(model_path)

    print(f"Input prompt: {prompt}")
    
    # Tokenize input
    input_tokens = model.tokenizer(
        [prompt],
        return_tensors="pt", # AirLLM uses PT-like interface for tokenizer
        return_attention_mask=False,
        truncation=True,
        max_length=128,
        padding=False
    )

    # Generate output
    print("Generating...")
    # MLX version of generate might have different signature, but AirLLM tries to keep it consistent
    generation_output = model.generate(
        input_tokens['input_ids'],
        max_new_tokens=50,
        use_cache=True,
    )

    # Decode and print results
    output = model.tokenizer.decode(generation_output[0], skip_special_tokens=True)
    print("\n--- Model Output ---")
    print(output)
    print("--------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AirLLM MLX Inference Example")
    parser.add_argument("--model_path", type=str, required=True, help="Path or Hub ID to the AirLLM sharded model")
    parser.add_argument("--prompt", type=str, default="Explain the theory of relativity in one sentence.", help="Prompt for inference")
    
    args = parser.parse_args()
    
    try:
        run_inference(args.model_path, args.prompt)
    except Exception as e:
        print(f"Error during inference: {e}")
