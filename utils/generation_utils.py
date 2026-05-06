import torch
import numpy as np


def generate(model, tokenizer, text: str, device: str, max_new_tokens: int = 20) -> str:
    """Run greedy decoding and return only the newly generated tokens as a string.

    Args:
        model:           HuggingFace CausalLM (already on device).
        tokenizer:       Matching tokenizer.
        text:            Full prompt string (already formatted with chat template).
        device:          "cuda" / "cpu" — used for input tensors.
        max_new_tokens:  How many tokens to generate.

    Returns:
        Generated text (prompt stripped).
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy — deterministic, good for evals
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated part
    new_ids = output_ids[0, input_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def extract_representation(
    model,
    tokenizer,
    text: str,
    device: str,
    num_final_tokens: int = 1,
    layer_idx: int = -1,
) -> np.ndarray:
    """Extract the hidden-state representation at a given layer.

    Uses a forward hook so it works with any HuggingFace decoder model
    (Llama 3, Mistral, Gemma, …) without modifying the model.

    Args:
        model:             HuggingFace CausalLM (already on device).
        tokenizer:         Matching tokenizer.
        text:              Full prompt string (already formatted).
        device:            "cuda" / "cpu".
        num_final_tokens:  How many trailing token representations to return.
                           1  → shape (hidden_size,)
                           k  → shape (k, hidden_size)
        layer_idx:         Which transformer layer to probe (0-indexed).
                           -1 means the last layer.

    Returns:
        NumPy float32 array of shape (hidden_size,) or (k, hidden_size).
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Resolve layer index
    # Llama 3 stores layers at model.model.layers
    layers = _get_layers(model)
    n_layers = len(layers)
    if layer_idx < 0:
        layer_idx = n_layers + layer_idx   # e.g. -1 → last layer

    captured = {}

    def hook_fn(module, input, output):
        # output is (hidden_states, ...) for decoder blocks
        hs = output[0] if isinstance(output, tuple) else output
        # Shape: (batch=1, seq_len, hidden_size)
        captured["hs"] = (
            hs[0, -num_final_tokens:, :]
            .detach()
            .cpu()
            .float()
            .numpy()
        )

    handle = layers[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(**inputs, output_hidden_states=False)
    finally:
        handle.remove()

    rep = captured["hs"]              # shape: (num_final_tokens, hidden_size)
    if num_final_tokens == 1:
        rep = rep[0]                  # flatten to (hidden_size,)
    return rep


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _get_layers(model):
    """Return the list of transformer decoder blocks for common HF architectures."""
    # Llama / Mistral / Gemma: model.model.layers
    nested = getattr(model, "model", None)
    if nested is not None:
        for attr in ("layers", "blocks", "h"):
            candidate = getattr(nested, attr, None)
            if candidate is not None and hasattr(candidate, "__getitem__"):
                return candidate

    # GPT-2: model.transformer.h
    transformer = getattr(model, "transformer", None)
    if transformer is not None:
        for attr in ("h", "layers", "blocks"):
            candidate = getattr(transformer, attr, None)
            if candidate is not None and hasattr(candidate, "__getitem__"):
                return candidate

    # TransformerLens: model.blocks
    for attr in ("blocks", "layers", "h"):
        candidate = getattr(model, attr, None)
        if candidate is not None and hasattr(candidate, "__getitem__"):
            return candidate

    raise RuntimeError(
        "Cannot find transformer layer list on model. "
        "Add a manual case to _get_layers() in generation_utils.py."
    )