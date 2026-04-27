"""Prompt parsing utilities."""
from __future__ import annotations

def parse_prompt_tags(prompt: str) -> list[str]:
    """Split a comma-style prompt into normalized tags."""
    return [p.strip().lower() for p in prompt.split(",") if p.strip()]
