"""
Example: High-efficiency inference using AirLLM.

AirLLM allows running large models on memory-constrained GPUs (e.g., 70B on 4GB)
by loading model layers one-by-one.

Usage:
    python examples/inference/airllm_inference.py --model_path "your-username/lfm-model-AirLLM"
"""

import argparse
from airllm import AutoModel

def run_inference(model_path: str, prompt: str, compression: str = None):
    print(f"Loading AirLLM model from: {model_path}")
    print(f"Compression: {compression or 'none'}")
    
    # Initialize the AirLLM model
    # Note: If it's a Hub ID, it will download and shard it if not already done.
    # If it's a local path to a sharded model (from our export), it will load it directly.
    model = AutoModel.from_pretrained(
        model_path,
        compression=compression,
        # profiling_mode=True,
    )

    print(f"Input prompt: {prompt}")
    
    # Tokenize input
    input_tokens = model.tokenizer(
        [prompt],
        return_tensors="pt",
        return_attention_mask=False,
        truncation=True,
        max_length=128,
        padding=False
    )

    # Generate output
    print("Generating...")
    generation_output = model.generate(
        input_tokens['input_ids'].cuda(),
        max_new_tokens=50,
        use_cache=True,
        return_dict_in_generate=True
    )

    # Decode and print results
    output = model.tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
    print("\n--- Model Output ---")
    print(output)
    print("--------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AirLLM Inference Example")
    parser.add_argument("--model_path", type=str, required=True, help="Path or Hub ID to the AirLLM sharded model")
    parser.add_argument("--prompt", type=str, default="What are the three laws of robotics?", help="Prompt for inference")
    parser.add_argument("--compression", type=str, choices=["4bit", "8bit"], default=None, help="Weight compression")
    
    args = parser.parse_args()
    
    try:
        run_inference(args.model_path, args.prompt, args.compression)
    except Exception as e:
        print(f"Error during inference: {e}")
        print("Make sure you have 'airllm' and 'bitsandbytes' installed.")
