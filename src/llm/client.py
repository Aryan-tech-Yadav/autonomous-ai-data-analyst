import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass
class LLMConfig:
    """
    Common configuration for an LLM provider.
    """

    provider: str
    api_key: str
    model: str
    base_url: str | None = None


def get_nvidia_config() -> LLMConfig:
    """
    Load NVIDIA Nemotron configuration.
    """

    api_key = os.getenv("NVIDIA_API_KEY")
    model = os.getenv("NVIDIA_MODEL")
    base_url = os.getenv(
        "NVIDIA_BASE_URL",
        "https://integrate.api.nvidia.com/v1",
    )

    if not api_key:
        raise ValueError(
            "NVIDIA_API_KEY is not configured."
        )

    if not model:
        raise ValueError(
            "NVIDIA_MODEL is not configured."
        )

    return LLMConfig(
        provider="nvidia",
        api_key=api_key,
        model=model,
        base_url=base_url,
    )


def get_anthropic_config() -> LLMConfig:
    """
    Load Anthropic Claude configuration.
    """

    api_key = os.getenv("ANTHROPIC_API_KEY")
    model = os.getenv("ANTHROPIC_MODEL")

    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is not configured."
        )

    if not model:
        raise ValueError(
            "ANTHROPIC_MODEL is not configured."
        )

    return LLMConfig(
        provider="anthropic",
        api_key=api_key,
        model=model,
    )
