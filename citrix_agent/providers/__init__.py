from .base import CallResult, ModelTurn, Provider, ToolCall


def make_provider(name: str, **kw) -> Provider:
    kw = {k: v for k, v in kw.items() if v is not None}
    if name == "claude":
        from .claude import ClaudeProvider
        return ClaudeProvider(**{k: v for k, v in kw.items() if k in ("model", "effort", "use_fallbacks")})
    if name == "gemini":
        from .gemini import GeminiProvider
        return GeminiProvider(**{k: v for k, v in kw.items() if k in ("model", "environment", "session_url")})
    raise ValueError(f"Unknown provider {name!r}")


__all__ = ["CallResult", "ModelTurn", "Provider", "ToolCall", "make_provider"]
