from dataclasses import dataclass
from enum import Enum
import base64
import re
from typing import Union
from src.preprocessor import ContentPreprocessor, PreprocessedContent, PreprocessedText, PreprocessedImage, PreprocessedVideo
from src.model import ContentModerationModel, ModelPrediction
from src.risk_classifier import RiskClassifier, PolicyClassification, RiskLevel


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
    """Result of content moderation"""
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


def _get_model_category(prediction_class: type) -> str:
    """
    Extract model category name from prediction class name.

    Args:
        prediction_class: The ModelPrediction subclass

    Returns:
        Category name (e.g., "hate_speech" from "HateSpeechPrediction")
    """
    class_name = prediction_class.__name__
    # Remove "Prediction" suffix and convert to snake_case
    category = class_name.replace("Prediction", "")
    # Convert CamelCase to snake_case
    return re.sub(r'(?<!^)(?=[A-Z])', '_', category).lower()


def _classify_policies(
        model_predictions: dict[str, Union[ModelPrediction, list[ModelPrediction]]]) -> PolicyClassification:
    """
    Classify policies from model predictions.

    For video content from a single model, uses the maximum scores across all frames to catch any harmful content.
    For text/image content or multiple models, uses the maximum average score across all models.

    Args:
        model_predictions: Dictionary mapping category to prediction(s)

    Returns:
        PolicyClassification with risk levels per category

    Raises:
        KeyError: If any expected category is missing from predictions
    """
    classifications = {}
    for category, prediction in model_predictions.items():
        if isinstance(prediction, list):
            # Check if this is video frames (same model_name) or multiple models (different model_names)
            model_names = set(p.model_name for p in prediction)
            if len(model_names) == 1:
                # Single model with video frames - use max score across frames
                max_scores: dict[str, float] = {}
                for frame_prediction in prediction:
                    prediction_dict = frame_prediction.to_dict()
                    for key, value in prediction_dict.items():
                        max_scores[key] = max(max_scores.get(key, 0.0), value)
                max_score = max(max_scores.values())
                classifications[category] = RiskClassifier.classify_score(max_score)
            else:
                # Multiple models for same category - use max average score across models
                max_avg_score = 0.0
                for model_prediction in prediction:
                    prediction_dict = model_prediction.to_dict()
                    avg_score = sum(prediction_dict.values()) / len(prediction_dict)
                    max_avg_score = max(max_avg_score, avg_score)
                classifications[category] = RiskClassifier.classify_score(max_avg_score)
        else:
            # Single model, single prediction
            prediction_dict = prediction.to_dict()
            avg_score = sum(prediction_dict.values()) / len(prediction_dict)
            classifications[category] = RiskClassifier.classify_score(avg_score)
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
        str, Union[ModelPrediction, list[ModelPrediction]]]:
        """
        Run all models on preprocessed content.

        Args:
            preprocessed_content: Preprocessed content from ContentPreprocessor

        Returns:
            Dictionary mapping model category names to their predictions (single or list for video)
        """
        predictions = {}
        for model in self.models:
            prediction: Union[ModelPrediction, list[ModelPrediction]] = await _predict_by_modality(model, preprocessed_content)
            # Extract category from prediction(s)
            if isinstance(prediction, list):
                # For video: get category from first frame prediction
                prediction_category: str = prediction[0].get_category()
            else:
                # For text/image: get category directly
                prediction_category: str = prediction.get_category()
            
            # If category already exists, convert to list of predictions
            if prediction_category in predictions:
                existing = predictions[prediction_category]
                if isinstance(existing, list):
                    existing.append(prediction)
                else:
                    predictions[prediction_category] = [existing, prediction]
            else:
                predictions[prediction_category] = prediction
        return predictions
