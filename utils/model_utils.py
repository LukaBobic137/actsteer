import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_from_tl_name(model_name: str, device: str = "cuda", cache_dir: str = None):
    """Load a HuggingFace model and tokenizer.

    The function name keeps backward compatibility with the original codebase
    (which used TransformerLens naming), but internally uses HuggingFace
    directly — works well with Llama 3.

    Args:
        model_name:  HuggingFace model ID, e.g.
                     "meta-llama/Meta-Llama-3-8B-Instruct"
                     "meta-llama/Meta-Llama-3-70B-Instruct"
        device:      "cuda", "cuda:0", "cpu", …
        cache_dir:   Optional path to HF cache directory.

    Returns:
        (model, tokenizer)
    """
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        use_fast=True,
    )

    # Llama 3 tokenizer has no pad token by default — set it to eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,   # fp16 saves VRAM; use bfloat16 on Ampere+
        device_map="auto",           # handles multi-GPU / CPU offload automatically
    )
    model.eval()

    return model, tokenizer