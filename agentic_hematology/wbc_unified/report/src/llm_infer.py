"""Batch report generation with Qwen3.5 + optional LoRA (HF backend)."""
from __future__ import annotations

from pathlib import Path
from typing import Any




def load_model_and_tokenizer(
    model_path: str | Path,
    adapter_path: str | Path | None = None,
):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    model_path = Path(model_path)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    if adapter_path and Path(adapter_path).is_dir():
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    model.eval()
    return model, tok


def encode_messages(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    import sys

    from .paths import REPO_ROOT

    verl_root = REPO_ROOT / "third_party" / "verl"
    if str(verl_root) not in sys.path:
        sys.path.insert(0, str(verl_root))
    # from verl.utils.chat_template import apply_chat_template
    # from verl.utils.tokenizer import normalize_token_ids
    from verl.utils.tokenizer.chat_template import apply_chat_template
    from verl.utils.tokenizer import normalize_token_ids

    return normalize_token_ids(
        apply_chat_template(
            tokenizer,
            messages,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=False,
        )
    )





def generate_from_messages(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int = 768,
    temperature: float = 0.0,
    repetition_penalty: float = 1.08,
) -> str:
    import torch

    prompt_ids = encode_messages(tokenizer, messages)
    inp = torch.tensor([prompt_ids], device=model.device)
    gen_kw: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        gen_kw.update(do_sample=True, temperature=temperature, repetition_penalty=repetition_penalty)
    else:
        gen_kw["do_sample"] = False

    with torch.no_grad():
        out = model.generate(inp, **gen_kw)
    return tokenizer.decode(out[0, len(prompt_ids) :], skip_special_tokens=True)




