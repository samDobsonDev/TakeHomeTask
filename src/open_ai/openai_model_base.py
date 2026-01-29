from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from pydantic import BaseModel
from src.model import ContentModerationModel, ModelPrediction
from src.preprocessor import PreprocessedText, PreprocessedImage, PreprocessedVideo, PreprocessedContent
from src.open_ai.openai_client import OpenAIClient

# Type variables for OpenAI content moderation models
# PredictionType: The final typed prediction output (e.g., HateSpeechPrediction)
# ResponseType: The OpenAI structured response format (e.g., HateSpeechScores)
PredictionType = TypeVar('PredictionType', bound=ModelPrediction)
ResponseType = TypeVar('ResponseType', bound=BaseModel)

class OpenAIContentModerationModel(ContentModerationModel[PredictionType], ABC, Generic[PredictionType, ResponseType]):
    """
    Base class for OpenAI-powered content moderation models.

    Handles the common logic for calling OpenAI and converting responses to predictions.
    Subclasses define model configuration and response format.
    """

    def __init__(self, api_key: str = None):
        """Initialize with OpenAI client"""
        self.client = OpenAIClient(api_key=api_key)

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the OpenAI model name to use"""
        pass

    @abstractmethod
    def get_text_prompt(self) -> str:
        """Return the prompt for text analysis"""
        pass

    @abstractmethod
    def get_image_prompt(self) -> str:
        """Return the prompt for image analysis"""
        pass

    @abstractmethod
    def get_response_format(self) -> type[ResponseType]:
        """Return the Pydantic model for response format"""
        pass

    @abstractmethod
    def _response_to_prediction(self, input_data: PreprocessedContent, response: ResponseType) -> PredictionType:
        """Convert OpenAI response to prediction dataclass"""
        pass

    async def predict_text(self, input_data: PreprocessedText) -> PredictionType:
        """Analyze text using OpenAI"""
        response = await self.client.analyze_text(
            text=input_data.original_text,
            model=self.get_model_name(),
            prompt=self.get_text_prompt(),
            response_format=self.get_response_format()
        )
        return self._response_to_prediction(input_data, response)

    async def predict_image(self, input_data: PreprocessedImage) -> PredictionType:
        """Analyze image using OpenAI"""
        response = await self.client.analyze_image(
            image_bytes=input_data.original_bytes,
            model=self.get_model_name(),
            prompt=self.get_image_prompt(),
            response_format=self.get_response_format()
        )
        return self._response_to_prediction(input_data, response)

    async def predict_video(self, input_data: PreprocessedVideo) -> list[PredictionType]:
        """Analyze video using OpenAI, returning per-frame predictions"""
        frames = [frame.original_bytes for frame in input_data.frames]
        responses = await self.client.analyze_video(
            frames=frames,
            model=self.get_model_name(),
            prompt=self.get_image_prompt(),
            response_format=self.get_response_format()
        )
        # Convert each response to a prediction
        predictions = []
        for i, response in enumerate(responses):
            prediction = self._response_to_prediction(input_data.frames[i], response)
            predictions.append(prediction)
        return predictions