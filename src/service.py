from dataclasses import dataclass
from enum import Enum
import base64
import asyncio
import logging
from typing import Union
from src.preprocessor import ContentPreprocessor, PreprocessedContent, PreprocessedText, PreprocessedImage, PreprocessedVideo
from src.model import ContentModerationModel, ModelPrediction, Category
from src.risk_classifier import RiskClassifier, PolicyClassification, RiskLevel
from src.score_calculator import ScoreCalculator

logger = logging.getLogger(__name__)


class Modality(Enum):
    """Content modality types"""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"


class PredictionType(Enum):
    """Prediction type"""
    POLICY = "policy"

@dataclass
class ModerationRequest:
    """Request to moderate content"""
    content: str | list[str]
    modality: Modality
    customer: str
    prediction_type: PredictionType = PredictionType.POLICY


@dataclass
class ModerationResult:
    """
    Result of content moderation containing both policy classification and raw predictions.
    
    Attributes:
        policy_classification: Final risk levels per category (high/medium/low).
                              Aggregates across all models by taking the maximum risk.
        model_predictions: Raw predictions from all models, organized by category.
                          Keys are Category enum values (e.g., Category.HATE_SPEECH).
                          Each category maps to either:
                          - A single ModelPrediction (text/image)
                          - A list of ModelPredictions (video frames)
                          This preserves individual model predictions for detailed analysis.
    
    Example:
        ModerationResult(
            policy_classification=PolicyClassification({
                Category.VIOLENCE: RiskLevel.HIGH,
                Category.HATE_SPEECH: RiskLevel.LOW
            }),
            model_predictions={
                Category.VIOLENCE: [frame1, frame2, frame3],  # Video frames
                Category.HATE_SPEECH: prediction_obj  # Single prediction
            }
        )
    """
    policy_classification: PolicyClassification
    model_predictions: dict[str, ModelPrediction | list[ModelPrediction]]


def _decode_content(content: str | list[str], modality: Modality) -> str | bytes | list[bytes]:
    """
    Decode content from base64 format.

    Args:
        content: Base64 encoded content (str for text/image, list[str] for video frames)
        modality: Content modality type

    Returns:
        Decoded content in appropriate format for the modality

    Raises:
        base64.binascii.Error: If base64 decoding fails
    """
    if modality == Modality.TEXT:
        # Text is not base64 encoded
        return content
    elif modality == Modality.IMAGE:
        # Decode single image from base64
        try:
            image_bytes = base64.b64decode(content)
            return image_bytes
        except Exception as e:
            raise ValueError(f"Failed to decode image from base64: {str(e)}")
    elif modality == Modality.VIDEO:
        # Decode video frames from list of base64 strings
        try:
            if not isinstance(content, list):
                raise ValueError("Video content must be a list of base64 strings")
            frames = []
            for frame_b64 in content:
                frame_bytes = base64.b64decode(frame_b64)
                frames.append(frame_bytes)
            return frames
        except Exception as e:
            raise ValueError(f"Failed to decode video from base64: {str(e)}")
    else:
        raise ValueError(f"Unknown modality: {modality}")


async def _predict_by_modality(model: ContentModerationModel, content: PreprocessedContent) -> Union[ModelPrediction, list[ModelPrediction]]:
    """
    Call the appropriate predict method on the model based on content type.

    Args:
        model: ContentModerationModel instance
        content: PreprocessedContent instance

    Returns:
        ModelPrediction from the model
    """
    if isinstance(content, PreprocessedText):
        return await model.predict_text(content)
    elif isinstance(content, PreprocessedImage):
        return await model.predict_image(content)
    elif isinstance(content, PreprocessedVideo):
        return await model.predict_video(content)
    else:
        raise ValueError(f"Unknown content type: {type(content)}")


def _get_prediction_category(prediction: Union[ModelPrediction, list[ModelPrediction]]) -> Category:
    """
    Extract the category from a prediction (single or list of frames).
    
    Args:
        prediction: Either a single ModelPrediction or a list of ModelPredictions (video frames)
    
    Returns:
        Category enum value from the prediction
    """
    if isinstance(prediction, list):
        return prediction[0].get_category()
    return prediction.get_category()


def _classify_policies(model_predictions: dict[str, list[Union[ModelPrediction, list[ModelPrediction]]]]) -> PolicyClassification:
    """
    Classify policies from model predictions.

    Args:
        model_predictions: Maps each category to a list of predictions from all models.
                          Each prediction can be a single ModelPrediction or a list of ModelPredictions (video).
    
    Uses maximum score across all frames and models to be on the safe side.
    """
    classifications = {}
    for category, predictions_list in model_predictions.items():
        max_avg_score = max(
            (ScoreCalculator.compute_average_score(prediction) for prediction in predictions_list),
            default=0.0
        )
        classifications[category] = RiskClassifier.classify_score(max_avg_score)
    return PolicyClassification(classifications=classifications)


class ContentModerationService:
    """
    Service that orchestrates content moderation.

    Coordinates preprocessing, model predictions, and risk classification
    without knowing about specific implementations.
    """

    def __init__(self, preprocessors: dict[str, ContentPreprocessor], models: list[ContentModerationModel]):
        """
        Initialize the service with preprocessors and models.

        Args:
            preprocessors: Dictionary mapping modality names to preprocessor instances
                          (e.g., {"text": TextPreprocessor(), "image": ImagePreprocessor()})
            models: List of content moderation models to run predictions with
        """
        self.preprocessors = preprocessors
        self.models = models

    async def moderate(self, request: ModerationRequest) -> ModerationResult:
        """
        Moderate content and return classification result.

        Args:
            request: ModerationRequest with content and metadata

        Returns:
            ModerationResult with policy classification and individual predictions

        Raises:
            ValueError: If modality is not supported
            base64.binascii.Error: If base64 decoding fails
        """
        if request.modality.value not in self.preprocessors:
            raise ValueError(f"Unsupported modality: {request.modality.value}")
        decoded_content = _decode_content(request.content, request.modality)
        preprocessed_content: PreprocessedContent = self.preprocessors[request.modality.value].preprocess(decoded_content)
        model_predictions = await self._run_all_models(preprocessed_content)
        policy_classification: PolicyClassification = _classify_policies(model_predictions)
        return ModerationResult(
            policy_classification=policy_classification,
            model_predictions=model_predictions
        )

    async def _run_all_models(self, preprocessed_content: PreprocessedContent) -> dict[
        str, list[Union[ModelPrediction, list[ModelPrediction]]]]:
        """
        Run all models concurrently on preprocessed content using asyncio.gather().

        Returns a dict mapping category to a list of predictions from all models.
        Each prediction can be:
        - A single ModelPrediction (for text/image)
        - A list of ModelPredictions (for video)

        This creates a normalized structure: always a list per category, containing 
        predictions from each model that predicted that category.
        """
        tasks = [
            _predict_by_modality(model, preprocessed_content)
            for model in self.models
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Process results and organize by category
        predictions_by_category = {}
        for model, result in zip(self.models, results):
            # Handle model failures gracefully
            if isinstance(result, Exception):
                logger.error(f"Model {model.name} failed with error: {result}")
                continue
            try:
                category = _get_prediction_category(result)
                predictions_by_category.setdefault(category, []).append(result)
            except Exception as e:
                logger.error(f"Failed to process prediction from model {model.name}: {e}")
                continue
        return predictions_by_category