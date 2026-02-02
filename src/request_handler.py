import json
from dataclasses import dataclass, field
from typing import Union
from src.risk_classifier import RiskClassifier, RiskLevel
from src.score_calculator import ScoreCalculator
from src.service import ContentModerationService, ModerationRequest, Modality, ModerationResult
from src.preprocessor import TextPreprocessor, ImagePreprocessor, VideoPreprocessor, ContentPreprocessor
from src.model import RandomHateSpeechModel, RandomSexualModel, RandomViolenceModel, ContentModerationModel, ModelPrediction

@dataclass
class FrameResult:
    """Result for a single video frame"""
    frame: int
    risk_level: RiskLevel
    scores: dict[str, float]


@dataclass
class ModelResult:
    """Base class for model prediction results"""
    model_name: str
    risk_level: RiskLevel


@dataclass
class TextImageModelResult(ModelResult):
    """Result for text/image model prediction"""
    scores: dict[str, float]


@dataclass
class VideoModelResult(ModelResult):
    """Result for video model prediction"""
    frames: list[FrameResult]


@dataclass
class CategoryResult:
    """Result for a content category (single model = list with one item)"""
    risk_level: RiskLevel
    models: list[Union[TextImageModelResult, VideoModelResult]]


@dataclass
class ModerationResponse:
    """Successful moderation response"""
    results: dict[str, CategoryResult]


@dataclass
class ErrorResponse:
    """Error response"""
    error: str
    status_code: int


class ServiceContainer:
    """
    Dependency injection container for all service dependencies.

    Manages instantiation and lifecycle of preprocessors, models, and the service.
    """

    def __init__(self, preprocessors: dict[str, ContentPreprocessor] = None, models: list[ContentModerationModel] = None):
        """
        Initialize the container and all dependencies.
        
        Args:
            preprocessors: Optional dictionary of preprocessors. If None, defaults are created.
            models: Optional list of models. If None, defaults are created.
        """
        self._preprocessors = preprocessors
        self._models = models
        self._service = None

    @property
    def preprocessors(self) -> dict[str, ContentPreprocessor]:
        """Lazy-load preprocessors"""
        if self._preprocessors is None:
            self._preprocessors = {
                "text": TextPreprocessor(),
                "image": ImagePreprocessor(),
                "video": VideoPreprocessor()
            }
        return self._preprocessors

    @property
    def models(self) -> list[ContentModerationModel]:
        """Lazy-load models"""
        if self._models is None:
            self._models = [
                RandomHateSpeechModel(),
                RandomSexualModel(),
                RandomViolenceModel(),
            ]
        return self._models

    @property
    def service(self) -> ContentModerationService:
        """Lazy-load service with dependencies"""
        if self._service is None:
            self._service = ContentModerationService(
                preprocessors=self.preprocessors,
                models=self.models
            )
        return self._service


def parse_request(request_data: dict) -> ModerationRequest:
    """
    Parse and validate request data.

    Args:
        request_data: Dictionary from JSON request

    Returns:
        ModerationRequest object

    Raises:
        ValueError: If validation fails
        KeyError: If required fields missing
    """
    # Validate required fields
    required_fields = ["content", "modality", "customer"]
    for field_name in required_fields:
        if field_name not in request_data:
            raise KeyError(field_name)
    try:
        modality = Modality(request_data["modality"])
    except ValueError:
        raise ValueError(
            f"Invalid modality: {request_data['modality']}. Must be one of: {', '.join(m.value for m in Modality)}")
    content: str = request_data["content"]
    if modality == Modality.VIDEO:
        if not isinstance(content, list):
            raise ValueError("Video content must be a list of base64-encoded frames")
        if len(content) == 0:
            raise ValueError("Video content cannot be empty")
        if not all(isinstance(frame, str) for frame in content):
            raise ValueError("All video frames must be base64-encoded strings")
    else:
        if not isinstance(content, str):
            raise ValueError(f"{modality.value.capitalize()} content must be a base64-encoded string")
        if not content:
            raise ValueError("Content cannot be empty")
    return ModerationRequest(
        content=content,
        modality=modality,
        customer=request_data["customer"]
    )


def format_success_response(moderation_result: ModerationResult) -> ModerationResponse:
    """
    Format successful moderation response.

    Input: ModerationResult with policy classifications and raw model predictions
    
    Output: ModerationResponse with organized results, reusing computed risk levels
    """
    results: dict[str, CategoryResult] = {}
    policy_classifications: dict[str, RiskLevel] = moderation_result.policy_classification.classifications
    for category, model_predictions in moderation_result.model_predictions.items():
        category_risk_level: RiskLevel = policy_classifications[category]
        models_results: list[TextImageModelResult | VideoModelResult] = [_build_model_result(prediction) for prediction in model_predictions]
        results[category] = CategoryResult(
            risk_level=category_risk_level,
            models=models_results
        )
    return ModerationResponse(results=results)


def _build_model_result(prediction: Union[ModelPrediction, list[ModelPrediction]]) -> Union[TextImageModelResult, VideoModelResult]:
    """
    Build a model result from either a single prediction or list of frames.
    
    Args:
        prediction: Either a ModelPrediction (text/image) or list[ModelPrediction] (video)
    
    Returns:
        TextImageModelResult for text/image, VideoModelResult for video
    """
    if isinstance(prediction, list):
        # Video: list of frames
        frame_results = [
            FrameResult(
                frame=idx,
                risk_level=RiskClassifier.classify_score(ScoreCalculator.compute_average_score(frame)),
                scores=frame.to_dict()
            )
            for idx, frame in enumerate(prediction)
        ]
        model_avg_score = ScoreCalculator.compute_average_score(prediction)
        model_risk_level = RiskClassifier.classify_score(model_avg_score)
        return VideoModelResult(
            model_name=prediction[0].model_name,
            risk_level=model_risk_level,
            frames=frame_results
        )
    else:
        # Text/image: single prediction
        model_avg_score = ScoreCalculator.compute_average_score(prediction)
        model_risk_level = RiskClassifier.classify_score(model_avg_score)
        return TextImageModelResult(
            model_name=prediction.model_name,
            risk_level=model_risk_level,
            scores=prediction.to_dict()
        )


def format_error_response(error_message: str, status_code: int) -> ErrorResponse:
    """
    Format error response.

    Args:
        error_message: Description of the error
        status_code: HTTP status code

    Returns:
        ErrorResponse object ready to be JSON serialized
    """
    return ErrorResponse(
        error=error_message,
        status_code=status_code
    )


class RequestHandler:
    """
    Mimics an HTTP request handler for content moderation API.

    Accepts JSON requests, delegates to the moderation service,
    and returns formatted JSON responses.
    """

    def __init__(self, container: ServiceContainer = None):
        """
        Initialize the request handler.

        Args:
            container: ServiceContainer with dependencies (uses default if None)
        """
        self.container = container or ServiceContainer()
        self.service = self.container.service

    async def handle_moderate_request(self, request_body: str) -> Union[ModerationResponse, ErrorResponse]:
        """
        Handle a content moderation request.

        Args:
            request_body: JSON string containing the request

        Returns:
            ModerationResponse or ErrorResponse object

        Raises:
            ValueError: If request format is invalid
            KeyError: If required fields are missing
        """
        try:
            request_data = json.loads(request_body)
            moderation_request: ModerationRequest = parse_request(request_data)
            result: ModerationResult = await self.service.moderate(moderation_request)
            return format_success_response(result)
        except ValueError as e:
            return format_error_response(str(e), 400)
        except KeyError as e:
            return format_error_response(f"Missing required field: {str(e)}", 400)
        except Exception as e:
            return format_error_response(f"Internal error: {str(e)}", 500)