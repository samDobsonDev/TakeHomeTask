import asyncio
import base64
import json
import os
import tempfile
from enum import Enum
from openai import OpenAI
from pydantic import BaseModel
from src.llm_client import LLMClient


class RequestModality(Enum):
    """Modality types for request building"""
    TEXT = "text"
    IMAGE = "image"


class OpenAIClient(LLMClient):
    """
    LLM client for interacting with OpenAI API.

    Provides methods for analyzing text, images, and videos using OpenAI's models.
    """

    def __init__(self, api_key: str = None):
        """Initialize the OpenAI client."""
        self.client = OpenAI(api_key=api_key)

    @staticmethod
    def build_request(prompt: str, modality: RequestModality, content: str | bytes) -> dict:
        """
        Build request payload for OpenAI API.

        Args:
            prompt: The analysis prompt
            modality: The content modality (TEXT or IMAGE)
            content: The content to analyze (str for text, bytes for image)

        Returns:
            Formatted request dictionary

        Raises:
            ValueError: If content is None
        """
        if content is None:
            raise ValueError("Content cannot be None")
        request_content = [
            {
                "type": "input_text",
                "text": prompt,
            }
        ]
        if modality == RequestModality.TEXT:
            # For text, append content to the prompt
            request_content[0]["text"] = f"{prompt}\n\nContent: {content}"
        elif modality == RequestModality.IMAGE:
            # For image, add as separate content block
            base64_image = base64.b64encode(content).decode('utf-8')
            request_content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{base64_image}",
            })
        return {
            "role": "user",
            "content": request_content,
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
        request = self.build_request(prompt, RequestModality.TEXT, text)
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
        request = self.build_request(prompt, RequestModality.IMAGE, image_bytes)
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
            request = self.build_request(prompt, RequestModality.IMAGE, frame_bytes)
            response = self.client.responses.parse(
                model=model,
                input=[request],
                text_format=response_format,
            )
            results.append(response.output_parsed)
        return results