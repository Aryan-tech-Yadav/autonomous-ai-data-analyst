from openai import OpenAI

from src.llm.client import get_nvidia_config


class NVIDIAClient:
    """
    Client for NVIDIA-hosted Nemotron models
    using the OpenAI-compatible API.
    """

    def __init__(self):
        self.config = get_nvidia_config()

        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """
        Send a chat request to the NVIDIA endpoint.
        """

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content or ""
