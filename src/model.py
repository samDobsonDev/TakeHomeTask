from abc import ABC, abstractmethod
from enum import StrEnum
import random
from dataclasses import dataclass, fields
from typing import Generic, TypeVar
from src.preprocessor import PreprocessedText, PreprocessedImage, PreprocessedVideo, PreprocessedContent


class Category(StrEnum):
    """Content moderation categories"""
    HATE_SPEECH = "hate_speech"
    SEXUAL = "sexual"
    VIOLENCE = "violence"


@dataclass
class ModelPrediction(ABC):
    """Base class for model predictions"""
    input_data: PreprocessedContent
    model_name: str

    @classmethod
    @abstractmethod
    def get_category(cls) -> Category:
        """Return the category for this prediction"""
        pass

    def to_dict(self) -> dict[str, float]:
        """Convert prediction to dictionary format"""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name not in ('input_data', 'model_name')}

PredictionType = TypeVar('PredictionType', bound=ModelPrediction)

@dataclass
class HateSpeechPrediction(ModelPrediction):
    """Prediction result from hate speech model"""
    toxicity: float
    severe_toxicity: float
    obscene: float
    insult: float
    identity_attack: float
    threat: float

    @classmethod
    def get_category(cls) -> Category:
        return Category.HATE_SPEECH


@dataclass
class SexualPrediction(ModelPrediction):
    """Prediction result from sexual content model"""
    sexual_explicit: float
    adult_content: float
    adult_toys: float

    @classmethod
    def get_category(cls) -> Category:
        return Category.SEXUAL


@dataclass
class ViolencePrediction(ModelPrediction):
    """Prediction result from violence model"""
    violence: float
    firearm: float
    knife: float

    @classmethod
    def get_category(cls) -> Category:
        return Category.VIOLENCE


class ContentModerationModel(ABC, Generic[PredictionType]):
    """
    Abstract base class for content moderation models.

    Each model evaluates content across different modalities (text, image, video)
    and returns raw metric scores.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the model - must be defined by subclasses"""
        ...

    def __init_subclass__(cls, **kwargs):
        """Validate that the declared PredictionType matches method return types"""
        super().__init_subclass__(**kwargs)
        # Only validate concrete implementations, not abstract intermediate classes
        # A concrete class is one that doesn't have any abstract methods remaining
        abstract_methods = getattr(cls, '__abstractmethods__', set())
        if abstract_methods:
            # This is still abstract, skip validation
            return
        # Get the declared PredictionType from the generic
        if hasattr(cls, '__orig_bases__'):
            for base in cls.__orig_bases__:
                if hasattr(base, '__args__') and base.__args__:
                    declared_type = base.__args__[0]
                    # Only validate if declared_type is a concrete class, not a TypeVar
                    from typing import TypeVar as TypeVarClass
                    if not isinstance(declared_type, TypeVarClass):
                        # This is a concrete type, validate it
                        # Check predict_text return type
                        if hasattr(cls, 'predict_text'):
                            predict_text_method = cls.predict_text
                            if hasattr(predict_text_method, '__annotations__'):
                                return_type = predict_text_method.__annotations__.get('return')
                                # Skip if return type is a TypeVar or matches declared type
                                if return_type and not isinstance(return_type, TypeVarClass) and return_type != declared_type:
                                    raise TypeError(
                                        f"{cls.__name__} declares {declared_type.__name__} "
                                        f"but predict_text returns {return_type.__name__ if hasattr(return_type, '__name__') else return_type}"
                                    )

    @abstractmethod
    async def predict_text(self, input_data: PreprocessedText) -> PredictionType:
        pass

    @abstractmethod
    async def predict_image(self, input_data: PreprocessedImage) -> PredictionType:
        pass

    async def predict_video(self, input_data: PreprocessedVideo) -> list[PredictionType]:
        """
        Default implementation: process each frame individually.

        Returns a prediction for each frame.
        """
        predictions = []
        for frame in input_data.frames:
            prediction = await self.predict_image(frame)
            predictions.append(prediction)
        return predictions


def _generate_random_scores(prediction_class: type[ModelPrediction]) -> dict[str, float]:
    """Generate random scores for all metrics in a prediction class"""
    metrics = [f.name for f in fields(prediction_class) if f.name not in ('input_data', 'model_name')]
    return {metric: random.random() for metric in metrics}


class RandomHateSpeechModel(ContentModerationModel[HateSpeechPrediction]):
    """Model for detecting hate speech and toxic content"""
    name = "RandomHateSpeechModel"

    async def predict_text(self, input_data: PreprocessedText) -> HateSpeechPrediction:
        scores = _generate_random_scores(HateSpeechPrediction)
        return HateSpeechPrediction(input_data=input_data, model_name=self.name, **scores)

    async def predict_image(self, input_data: PreprocessedImage) -> HateSpeechPrediction:
        scores = _generate_random_scores(HateSpeechPrediction)
        return HateSpeechPrediction(input_data=input_data, model_name=self.name, **scores)


class RandomSexualModel(ContentModerationModel[SexualPrediction]):
    """Model for detecting sexual content"""
    name = "RandomSexualModel"

    async def predict_text(self, input_data: PreprocessedText) -> SexualPrediction:
        scores = _generate_random_scores(SexualPrediction)
        return SexualPrediction(input_data=input_data, model_name=self.name, **scores)

    async def predict_image(self, input_data: PreprocessedImage) -> SexualPrediction:
        scores = _generate_random_scores(SexualPrediction)
        return SexualPrediction(input_data=input_data, model_name=self.name, **scores)


class RandomViolenceModel(ContentModerationModel[ViolencePrediction]):
    """Model for detecting violent content"""
    name = "RandomViolenceModel"

    async def predict_text(self, input_data: PreprocessedText) -> ViolencePrediction:
        scores = _generate_random_scores(ViolencePrediction)
        return ViolencePrediction(input_data=input_data, model_name=self.name, **scores)

    async def predict_image(self, input_data: PreprocessedImage) -> ViolencePrediction:
        scores = _generate_random_scores(ViolencePrediction)
        return ViolencePrediction(input_data=input_data, model_name=self.name, **scores)