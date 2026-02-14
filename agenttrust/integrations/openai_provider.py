"""OpenAI / Anthropic provider wrapper for AgentTrust.

Provides a simple interface to call LLMs with AgentTrust calibration
built in. Requires the ``openai`` extra: ``pip install agenttrust[openai]``
"""

from __future__ import annotations

from typing import Any


def create_calibrated_fn(
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    base_url: str | None = None,
    system_prompt: str = "Answer concisely and accurately.",
    **kwargs: Any,
) -> Any:
    """Create a callable suitable for ``sample_consistency``.

    Returns a function ``fn(query: str) -> str`` that calls the OpenAI-compatible
    API. Works with OpenAI, Anthropic (via proxy), and any OpenAI-compatible
    endpoint.

    Args:
        model: Model name (e.g., "gpt-4o-mini", "claude-sonnet-4-20250514").
        api_key: API key. Falls back to OPENAI_API_KEY env var.
        base_url: Custom base URL for Anthropic or other providers.
        system_prompt: System prompt to use.
        **kwargs: Additional arguments passed to the API call.

    Returns:
        A callable ``fn(query: str) -> str``.

    Raises:
        ImportError: If the ``openai`` package is not installed.

    Example::

        >>> fn = create_calibrated_fn(model="gpt-4o-mini")  # doctest: +SKIP
        >>> from agenttrust import sample_consistency
        >>> result = sample_consistency(fn, "What is 2+2?")  # doctest: +SKIP
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package required. Install with: pip install agenttrust[openai]"
        )

    client_kwargs: dict[str, Any] = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)

    def fn(query: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 256),
        )
        return response.choices[0].message.content or ""

    return fn
