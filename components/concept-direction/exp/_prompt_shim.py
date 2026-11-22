"""Chat spelling resolved from the published prompt-configs component (cache-only)."""


def GemmaPromptConfig():  # noqa: N802 - matches the published definition's name
    """The Gemma chat-spelling config class, without importing the examples package."""
    from interpretune.hub.promptconfigs import resolve_prompt_config_class

    return resolve_prompt_config_class("speediedan/prompt-configs#GemmaPromptConfig")()
