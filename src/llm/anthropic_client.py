from anthropic import Anthropic

from src.llm.client import get_anthropic_config


class AnthropicClient:
    """
    Client for Anthropic Claude models.
    """

    def __init__(self):
        self.config = get_anthropic_config()

        self.client = Anthropic(
            api_key=self.config.api_key,
        )

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """
        Send a chat request to Claude.
        """

        system_message = None
        user_messages = []

        for message in messages:

            role = message.get("role")
            content = message.get("content", "")

            if role == "system":
                system_message = content

            else:
                user_messages.append(
                    {
                        "role": role,
                        "content": content,
                    }
                )

        request = {
            "model": self.config.model,
            "messages": user_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if system_message:
            request["system"] = system_message

        response = self.client.messages.create(
            **request
        )

        if not response.content:
            return ""

        return response.content[0].text
