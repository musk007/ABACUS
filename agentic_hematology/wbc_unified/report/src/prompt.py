"""Shared prompts. Single source of truth: data_preprocessing.report_protocol."""
from __future__ import annotations

from typing import Any

from data_preprocessing.report_prompt import SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT
from data_preprocessing.report_protocol import build_llm_messages


def summary_to_user_content(summary: dict[str, Any], template_md: str) -> str:
    return build_llm_messages(summary, template_md)[1]["content"]


def build_messages(
    summary: dict[str, Any],
    template_md: str,
    *,
    system_prompt: str | None = None,
    assistant_content: str | None = None,
) -> list[dict[str, str]]:
    """SFT triple or inference pair (no assistant if assistant_content is None)."""
    msgs = [dict(m) for m in build_llm_messages(summary, template_md)]
    if system_prompt is not None:
        msgs[0]["content"] = system_prompt
    if assistant_content is not None:
        msgs.append({"role": "assistant", "content": assistant_content})
    return msgs


def messages_for_inference(
    summary: dict[str, Any], template_md: str, system_prompt: str | None = None
) -> list[dict[str, str]]:
    """Chat layout aligned with verl SFT/GRPO (system merged into first user)."""
    from .verl_parquet import normalize_messages_for_verl

    return normalize_messages_for_verl(
        build_messages(summary, template_md, system_prompt=system_prompt)
    )