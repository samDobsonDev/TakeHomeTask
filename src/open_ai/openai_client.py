import base64
from openai import OpenAI
from pydantic import BaseModel
from src.llm_client import LLMClient


class OpenAIClient(LLMClient):
    """
    LLM client for interacting with OpenAI API.

    Provides methods for analyzing text, images, and videos using OpenAI's models.
    """

    def __init__(self, api_key: str = None):
        """Initialize the OpenAI client."""
        self.client = OpenAI(api_key=api_key)

    @staticmethod
    def _build_text_request(prompt: str, text: str) -> dict:
        """
        Build request payload for text content.

        Args:
            prompt: The analysis prompt
            text: The text content to analyze

        Returns:
            Formatted request dictionary

        Raises:
            ValueError: If text is None
        """
        if text is None:
            raise ValueError("Content cannot be None")
        return {
            "role": "user",
            "content": [{"type": "input_text", "text": f"{prompt}\n\nContent: {text}"}],
        }

    @staticmethod
    def _build_image_request(prompt: str, image_bytes: bytes) -> dict:
        """
        Build request payload for image content.

        Args:
            prompt: The analysis prompt
            image_bytes: The image bytes to analyze

        Returns:
            Formatted request dictionary

        Raises:
            ValueError: If image_bytes is None
        """
        if image_bytes is None:
            raise ValueError("Content cannot be None")
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        return {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image}"},
            ],
        }

    async def analyze_text(
        self,
        text: str,
        model: str,
        prompt: str,
        response_format: type[BaseModel]
    ) -> BaseModel:
        """
        Analyze text using OpenAI.

        Args:
            text: The text to analyze
            model: The model to use (e.g., "gpt-4o-2024-08-06")
            prompt: The prompt to send to the model
            response_format: Pydantic model for structured output

        Returns:
            Parsed response as specified by response_format
        """
        request = self._build_text_request(prompt, text)
        response = self.client.responses.parse(
            model=model,
            input=[request],
            text_format=response_format,
        )
        return response.output_parsed

    async def analyze_image(
        self,
        image_bytes: bytes,
        model: str,
        prompt: str,
        response_format: type[BaseModel]
    ) -> BaseModel:
        """
        Analyze image using OpenAI.

        Args:
            image_bytes: The image bytes to analyze
            model: The model to use
            prompt: The prompt to send to the model
            response_format: Pydantic model for structured output

        Returns:
            Parsed response as specified by response_format
        """
        request = self._build_image_request(prompt, image_bytes)
        response = self.client.responses.parse(
            model=model,
            input=[request],
            text_format=response_format,
        )
        return response.output_parsed

    async def analyze_video(
        self,
        frames: list[bytes],
        model: str,
        prompt: str,
        response_format: type[BaseModel]
    ) -> list[BaseModel]:
        """
        Analyze video frames using OpenAI.

        Args:
            frames: List of frame bytes to analyze
            model: The model to use
            prompt: The prompt to send to the model
            response_format: Pydantic model for structured output

        Returns:
            List of parsed responses (one per frame)
        """
        results = []
        for frame_bytes in frames:
            request = self._build_image_request(prompt, frame_bytes)
            response = self.client.responses.parse(
                model=model,
                input=[request],
                text_format=response_format,
            )
            results.append(response.output_parsed)
        return results