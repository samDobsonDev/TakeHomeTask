from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Type, get_args
from pydantic import BaseModel
from src.model import ContentModerationModel, ModelPrediction
from src.preprocessor import PreprocessedText, PreprocessedImage, PreprocessedVideo, PreprocessedContent
from src.open_ai.openai_client import OpenAIClient
from src.pydantic_generator import prediction_to_pydantic

# Type variable for OpenAI content moderation models
PredictionType = TypeVar('PredictionType', bound=ModelPrediction)


class OpenAIContentModerationModel(ContentModerationModel[PredictionType], ABC, Generic[PredictionType]):
    """
    Base class for OpenAI-powered content moderation models.

    Handles the common logic for calling OpenAI and converting responses to predictions.
    Subclasses only need to define:
    - name: The model name
    - get_model_name(): The OpenAI model to use
    - get_text_prompt(): Prompt for text analysis
    - get_image_prompt(): Prompt for image analysis

    The prediction class is automatically extracted from the generic type parameter,
    and the Pydantic response format is generated from it.
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

    @property
    def _prediction_class(self) -> Type[PredictionType]:
        """
        Extract the prediction class from the generic type parameter.

        For a class like OpenAIHateSpeechModel(OpenAIContentModerationModel[HateSpeechPrediction]),
        this returns HateSpeechPrediction.
        """
        orig_bases = getattr(self.__class__, '__orig_bases__', ())
        for base in orig_bases:
            args = get_args(base)
            if args and isinstance(args[0], type) and issubclass(args[0], ModelPrediction):
                return args[0]
        raise TypeError(f"{self.__class__.__name__} must specify a prediction type parameter")

    def get_response_format(self) -> Type[BaseModel]:
        """
        Return the Pydantic model for response format.

        Automatically generated from the prediction class to ensure consistency.
        """
        return prediction_to_pydantic(self._prediction_class)

    def _response_to_prediction(self, input_data: PreprocessedContent, response: BaseModel) -> PredictionType:
        """
        Convert OpenAI response to prediction dataclass.

        Default implementation maps all fields from the response to the prediction class.
        """
        scores = {field_name: getattr(response, field_name) for field_name in response.model_fields.keys()}
        return self._prediction_class(input_data=input_data, model_name=self.name, **scores)

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