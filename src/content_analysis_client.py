from abc import ABC, abstractmethod
from pydantic import BaseModel


class ContentAnalysisClient(ABC):
    """
    Abstract interface for LLM-based content analysis clients.

    Provides a contract for connecting to various LLM providers (OpenAI, Anthropic, etc.)
    to analyze text, images, and video content with structured output support.

    Implementations handle:
    - Authentication and API communication with LLM providers
    - Content encoding (text input, image/video byte handling)
    - Structured response parsing using Pydantic models
    - Batch processing for video frames
    """

    @abstractmethod
    async def analyze_text(
            self,
            text: str,
            model: str,
            prompt: str,
            response_format: type[BaseModel]
    ) -> BaseModel:
        """
        Analyze text content.

        Args:
            text: Raw text to analyze
            model: Model identifier to use for analysis
            prompt: Instructions for the model
            response_format: Pydantic model for structured output

        Returns:
            Structured response matching response_format
        """
        pass

    @abstractmethod
    async def analyze_image(
            self,
            image_bytes: bytes,
            model: str,
            prompt: str,
            response_format: type[BaseModel]
    ) -> BaseModel:
        """
        Analyze image content.

        Args:
            image_bytes: Raw image bytes to analyze
            model: Model identifier to use for analysis
            prompt: Instructions for the model
            response_format: Pydantic model for structured output

        Returns:
            Structured response matching response_format
        """
        pass

    @abstractmethod
    async def analyze_video(
            self,
            frames: list[bytes],
            model: str,
            prompt: str,
            response_format: type[BaseModel]
    ) -> list[BaseModel]:
        """
        Analyze video content frame by frame.

        Args:
            frames: List of frame bytes to analyze
            model: Model identifier to use for analysis
            prompt: Instructions for the model
            response_format: Pydantic model for structured output

        Returns:
            List of structured responses, one per frame
        """
        pass