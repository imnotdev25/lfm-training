"""
Example: Inference with TurboQuant KV Cache Quantization.

TurboQuant provides high-efficiency KV cache quantization (2.5-bit or 3.5-bit) 
with minimal quality loss by identifying and preserving outlier channels.

This example demonstrates how to use the TurboQuant metadata generated 
by lfm-trainer with different inference engines.
"""

import argparse
import json
import os

def turboquant_vllm_example(model_path: str):
    print("\n--- vLLM TurboQuant Example ---")
    print("To use TurboQuant with vLLM, you need the 'vllm-turboquant' fork.")
    print("The metadata file 'turboquant_metadata.json' should be in the model directory.")
    
    example_code = f"""
from vllm import LLM, SamplingParams

# vllm-turboquant automatically detects turboquant_metadata.json in the model folder
llm = LLM(
    model="{model_path}",
    kv_cache_dtype="turboquant25",  # or "turboquant35"
)

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)
outputs = llm.generate(["Explain quantum entanglement."], sampling_params)

for output in outputs:
    print(f"Generated: {{output.outputs[0].text}}")
"""
    print(example_code)

def turboquant_mlx_example(model_path: str):
    print("\n--- MLX TurboQuant Example ---")
    print("TurboQuant for MLX uses randomized Hadamard transforms and layer-adaptive compression.")
    
    example_code = f"""
from mlx_lm import load, generate
from turboquant_mlx import make_adaptive_cache, apply_patch

# 1. Load model and tokenizer
model, tokenizer = load("{model_path}")

# 2. Enable fused Metal kernels for TurboQuant
apply_patch()

# 3. Create TurboQuant adaptive cache
# bits=3 (TurboQuant 3-bit), fp16_layers=4 (keep first/last 4 layers in FP16)
cache = make_adaptive_cache(len(model.layers), bits=3, fp16_layers=4)

# 4. Generate
prompt = "What is the capital of France?"
response = generate(model, tokenizer, prompt=prompt, cache=cache, verbose=True)
"""
    print(example_code)

def main():
    parser = argparse.ArgumentParser(description="TurboQuant Inference Examples")
    parser.add_argument("--model_path", type=str, default="your-username/lfm-model-TurboQuant", 
                        help="Path or Hub ID to the TurboQuant model")
    
    args = parser.parse_args()
    
    print("TurboQuant KV Cache Quantization Inference")
    print("==========================================")
    
    turboquant_vllm_example(args.model_path)
    turboquant_mlx_example(args.model_path)

if __name__ == "__main__":
    main()
