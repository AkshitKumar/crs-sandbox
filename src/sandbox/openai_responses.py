"""Small helpers for OpenAI Responses API calls."""

from __future__ import annotations

from typing import Any


OPENAI_MAX_RETRIES = 7


def content_to_text(content: Any) -> str:
    """Extract text from strings, content blocks, or SDK response objects."""
    text = getattr(content, "text", None)
    if isinstance(text, str):
        return text

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            extracted = content_to_text(block)
            if extracted:
                parts.append(extracted)
        return "\n".join(parts).strip()

    if isinstance(content, dict):
        for key in ("text", "refusal", "content", "output_text"):
            value = content.get(key)
            if value is not None:
                extracted = content_to_text(value)
                if extracted:
                    return extracted
        return ""

    nested_content = getattr(content, "content", None)
    if nested_content is not None:
        extracted = content_to_text(nested_content)
        if extracted:
            return extracted

    return ""


def message_to_text(message: Any) -> str:
    """Extract customer-facing text from a LangChain/OpenAI message object."""
    text = content_to_text(message)
    return text if text else str(message)


def response_to_text(response: Any) -> str:
    """Extract final output text from an OpenAI Responses API response."""
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text:
        return output_text

    output = getattr(response, "output", None)
    text = content_to_text(output)
    return text if text else str(response)
