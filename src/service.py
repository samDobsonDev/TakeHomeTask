from dataclasses import dataclass
from enum import Enum
import base64
import re
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
    content: str  # base64 encoded for image/video, plain text for text
    modality: Modality
    customer: str
    prediction_type: PredictionType = PredictionType.POLICY


@dataclass
class ModerationResult:
    """Result of content moderation"""
    policy_classification: PolicyClassification
    model_predictions: dict[str, ModelPrediction]


def _decode_content(content: str, modality: Modality) -> str | bytes | list[bytes]:
    """
    Decode content from base64 format.

    Args:
        content: Base64 encoded content
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
        # Decode video frames from base64
        # Format: comma-separated base64 strings, or single base64 string for mock
        try:
            # For simplicity, assume single base64 string (will be expanded to frames by preprocessor)
            video_bytes = base64.b64decode(content)
            return [video_bytes]
        except Exception as e:
            raise ValueError(f"Failed to decode video from base64: {str(e)}")
    else:
        raise ValueError(f"Unknown modality: {modality}")


async def _predict_by_modality(model: ContentModerationModel, content: PreprocessedContent) -> ModelPrediction:
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


def _classify_policies(model_predictions: dict[str, ModelPrediction]) -> PolicyClassification:
    """
    Convert model predictions to policy classification.

    Args:
        model_predictions: Dictionary of model predictions

    Returns:
        PolicyClassification with risk levels for each category
    """
    classifications = {}
    for category, prediction in model_predictions.items():
        # Get raw scores from prediction
        scores: dict[str, float] = prediction.to_dict()
        # Average scores and classify to risk level
        avg_score: float = RiskClassifier.average_prediction_scores(scores)
        risk_level: RiskLevel = RiskClassifier.classify_score(avg_score)
        classifications[category] = risk_level
    return PolicyClassification(
        hate_speech=classifications.get("hate_speech"),
        sexual=classifications.get("sexual"),
        violence=classifications.get("violence")
    )


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
        # Validate request
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

    async def _run_all_models(self, preprocessed_content: PreprocessedContent) -> dict[str, ModelPrediction]:
        """
        Run all models on preprocessed content.

        Args:
            preprocessed_content: Preprocessed content from ContentPreprocessor

        Returns:
            Dictionary mapping model category names to their predictions
        """
        predictions = {}
        for model in self.models:
            # Get category name from model's prediction class
            prediction: ModelPrediction = await _predict_by_modality(model, preprocessed_content)
            prediction_category = prediction.get_category()
            predictions[prediction_category] = prediction
        return predictions

