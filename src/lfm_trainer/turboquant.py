"""
TurboQuant calibration and metadata generation for KV cache quantization.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional, Dict, List

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Constants from vllm-turboquant
TURBOQUANT_METADATA_VERSION = 1
TURBOQUANT_TRANSFORM_VERSION = "structured_hadamard_v1"
TURBOQUANT_GROUP_ALIGNMENT = 16
TURBOQUANT_OUTLIER_RATIOS = {
    "turboquant25": 0.25,
    "turboquant35": 0.50,
}

def get_turboquant_outlier_count(head_size: int, kv_cache_dtype: str) -> int:
    if head_size % TURBOQUANT_GROUP_ALIGNMENT != 0:
        raise ValueError(
            f"TurboQuant KV cache requires head_size to be a multiple of {TURBOQUANT_GROUP_ALIGNMENT}."
        )
    ratio = TURBOQUANT_OUTLIER_RATIOS[kv_cache_dtype]
    aligned = int(
        round(head_size * ratio / TURBOQUANT_GROUP_ALIGNMENT)
        * TURBOQUANT_GROUP_ALIGNMENT
    )
    if aligned <= 0 or aligned >= head_size:
        raise ValueError(
            f"Unsupported TurboQuant head_size {head_size} for {kv_cache_dtype}."
        )
    return aligned

class TurboQuantCalibrator:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        kv_cache_dtype: str = "turboquant25",
        max_seq_len: int = 512,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.kv_cache_dtype = kv_cache_dtype
        self.max_seq_len = max_seq_len
        self.stats = {}
        self.hooks = []

    def _get_hook(self, name: str):
        def hook(module, input, output):
            # output is (batch, seq_len, num_heads, head_size) or similar
            # We want to collect sum of squares per head per channel
            # Usually LLM attention output is (batch, seq_len, hidden_size)
            # but we hook the projection layers.
            
            # activation: (batch, seq_len, hidden_size)
            # We want to identify channels that are outliers.
            # In TurboQuant, we look at the output of K and V projections.
            
            # output: (batch, seq_len, num_heads * head_size)
            # Reshape to (batch, seq_len, num_heads, head_size)
            
            # For simplicity, let's assume standard Transformers layout
            # where hidden_size = num_heads * head_size.
            # We can find num_heads from the model config.
            
            num_heads = getattr(self.model.config, "num_key_value_heads", self.model.config.num_attention_heads)
            head_size = self.model.config.hidden_size // self.model.config.num_attention_heads
            
            # output might be (batch, seq_len, num_heads * head_size)
            # or if it's GQA/MQA, it might be different.
            
            act = output.detach().float()
            if len(act.shape) == 3:
                # (batch, seq_len, total_kv_dim)
                batch, seq, dim = act.shape
                act = act.view(batch, seq, num_heads, -1)
            
            # sum of squares: (num_heads, head_size)
            # We sum over batch and seq_len
            ssq = (act ** 2).sum(dim=(0, 1))
            
            if name not in self.stats:
                self.stats[name] = torch.zeros_like(ssq)
            self.stats[name] += ssq
            
        return hook

    def register_hooks(self):
        # We need to find the K and V projection layers.
        # This depends on the model architecture. 
        # For LFM/Llama/Mistral, it's usually self_attn.k_proj and self_attn.v_proj
        
        for name, module in self.model.named_modules():
            if name.endswith(".k_proj") or name.endswith(".v_proj"):
                logger.info(f"Registering TurboQuant calibration hook for {name}")
                self.hooks.append(module.register_forward_hook(self._get_hook(name)))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    @torch.no_grad()
    def calibrate(self, calibration_data: List[str]):
        self.model.eval()
        self.register_hooks()
        
        logger.info(f"Running TurboQuant calibration on {len(calibration_data)} samples...")
        for text in tqdm(calibration_data):
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=self.max_seq_len
            ).to(self.model.device)
            self.model(**inputs)
            
        self.remove_hooks()
        return self.compute_metadata()

    def compute_metadata(self) -> Dict[str, Any]:
        # head_size = self.model.config.hidden_size // self.model.config.num_attention_heads
        # Wait, LFM might have different head size.
        # Better use the actual shape of collected stats.
        
        # Determine head_size and model_name
        head_size = getattr(self.model.config, "head_dim", 
                            self.model.config.hidden_size // self.model.config.num_attention_heads)
        model_name = getattr(self.model.config, "_name_or_path", "model")

        metadata = {
            "version": TURBOQUANT_METADATA_VERSION,
            "recipe": self.kv_cache_dtype,
            "head_size": head_size,
            "model_name": model_name,
            "transform_version": TURBOQUANT_TRANSFORM_VERSION,
            "codebook_version": "lloyd_beta_v1",
            "layers": {}
        }
        
        # stats: { "layer.k_proj": (num_heads, head_size), ... }
        
        # We need to group K and V together for each layer.
        layer_names = set()
        for key in self.stats.keys():
            layer_name = key.rsplit(".", 1)[0]
            layer_names.add(layer_name)
            
        for layer in sorted(list(layer_names)):
            k_key = f"{layer}.k_proj"
            v_key = f"{layer}.v_proj"
            
            if k_key not in self.stats or v_key not in self.stats:
                continue
                
            k_stats = self.stats[k_key]
            v_stats = self.stats[v_key]
            
            curr_num_heads, curr_head_size = k_stats.shape
            outlier_count = get_turboquant_outlier_count(curr_head_size, self.kv_cache_dtype)
            
            def get_outliers(stats):
                # stats: (num_heads, head_size)
                # For each head, get top-k outlier indices
                outliers = []
                for head_idx in range(curr_num_heads):
                    head_stats = stats[head_idx]
                    _, indices = torch.topk(head_stats, outlier_count)
                    outliers.append(sorted(indices.tolist()))
                return outliers

            metadata["layers"][layer] = {
                "key_high_precision_indices": get_outliers(k_stats),
                "value_high_precision_indices": get_outliers(v_stats),
            }
            
        return metadata

def calibrate_and_save_turboquant(
    model: torch.nn.Module,
    tokenizer: Any,
    output_path: str,
    calibration_data: Optional[List[str]] = None,
    kv_cache_dtype: str = "turboquant25",
    max_prompts: int = 128,
    max_seq_len: int = 512,
):
    if calibration_data is None:
        # Fallback: use some default data or just random if not available
        # But usually we should have it.
        logger.warning("No calibration data provided for TurboQuant. Using empty list.")
        calibration_data = []
    
    # Cap the number of prompts
    calibration_data = calibration_data[:max_prompts]
    
    calibrator = TurboQuantCalibrator(
        model=model,
        tokenizer=tokenizer,
        kv_cache_dtype=kv_cache_dtype,
        max_seq_len=max_seq_len
    )
    
    metadata = calibrator.calibrate(calibration_data)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✅ TurboQuant metadata saved to {output_path}")
    return metadata
