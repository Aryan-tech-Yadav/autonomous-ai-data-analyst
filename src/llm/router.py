import time

from src.llm.anthropic_client import AnthropicClient
from src.llm.nvidia_client import NVIDIAClient


class LLMRouter:
    """
    Provider-independent LLM router.

    The rest of the application can use this router
    without knowing which LLM provider is being used.
    """

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self._nvidia = None
        self._anthropic = None

        self.max_retries = max_retries
        self.retry_delay = retry_delay

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

    def _get_client(self, provider: str):
        """
        Resolve the requested provider client.
        """

        if provider == "nvidia":
            return self._get_nvidia()

        if provider in {"anthropic", "claude"}:
            return self._get_anthropic()

        raise ValueError(
            f"Unsupported LLM provider: {provider}"
        )

    def generate(
        self,
        messages: list[dict],
        provider: str = "nvidia",
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """
        Generate a response using the selected provider.

        Temporary provider failures are retried automatically.
        """

        provider = provider.lower().strip()
        client = self._get_client(provider)

        last_error = None

        for attempt in range(
            self.max_retries + 1
        ):
            try:
                return client.generate(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            except Exception as exc:
                last_error = exc

                if attempt >= self.max_retries:
                    raise

                delay = (
                    self.retry_delay
                    * (2 ** attempt)
                )

                print(
                    f"LLM request failed for "
                    f"{provider} "
                    f"(attempt {attempt + 1}/"
                    f"{self.max_retries + 1}). "
                    f"Retrying in {delay:.1f}s...",
                    flush=True,
                )

                time.sleep(delay)

        raise last_error

    def available_providers(self) -> list[str]:
        """
        Return supported LLM providers.
        """

        return [
            "nvidia",
            "anthropic",
        ]
