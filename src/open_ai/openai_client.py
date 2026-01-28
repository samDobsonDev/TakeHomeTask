import base64
from openai import OpenAI
from pydantic import BaseModel
from src.llm_client import LLMClient


class OpenAIClient(LLMClient):
    """
    ContentAnalysisClient for interacting with OpenAI API.

    Provides methods for analyzing text, images, and videos using OpenAI's models.
    """

    def __init__(self, api_key: str = None):
        """Initialize the OpenAI client."""
        self.client = OpenAI(api_key=api_key)

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
        response = self.client.responses.parse(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"{prompt}\n\nContent: {text}",
                        }
                    ],
                }
            ],
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
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        response = self.client.responses.parse(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    ],
                }
            ],
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
        Analyze video frames using OpenAI with batch processing.

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
            base64_frame = base64.b64encode(frame_bytes).decode('utf-8')
            response = self.client.responses.parse(
                model=model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": prompt,
                            },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{base64_frame}",
                            },
                        ],
                    }
                ],
                text_format=response_format,
            )
            results.append(response.output_parsed)
        return results