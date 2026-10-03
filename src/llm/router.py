from src.llm.anthropic_client import AnthropicClient
from src.llm.nvidia_client import NVIDIAClient


class LLMRouter:
    """
    Provider-independent LLM router.

    The rest of the application can use this router
    without knowing which LLM provider is being used.
    """

    def __init__(self):
        self._nvidia = None
        self._anthropic = None

    def _get_nvidia(self) -> NVIDIAClient:
        """
        Create NVIDIA client lazily.
        """

        if self._nvidia is None:
            self._nvidia = NVIDIAClient()

        return self._nvidia

    def _get_anthropic(self) -> AnthropicClient:
        """
        Create Anthropic client lazily.
        """

        if self._anthropic is None:
            self._anthropic = AnthropicClient()

        return self._anthropic

    def generate(
        self,
        messages: list[dict],
        provider: str = "nvidia",
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """
        Generate a response using the selected provider.
        """

        provider = provider.lower().strip()

        if provider == "nvidia":
            client = self._get_nvidia()

        elif provider in {"anthropic", "claude"}:
            client = self._get_anthropic()

        else:
            raise ValueError(
                f"Unsupported LLM provider: {provider}"
            )

        return client.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def available_providers(self) -> list[str]:
        """
        Return supported LLM providers.
        """

        return [
            "nvidia",
            "anthropic",
        ]
