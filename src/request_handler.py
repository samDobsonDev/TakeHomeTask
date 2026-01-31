import json
from typing import Union
from src.risk_classifier import RiskClassifier
from src.service import ContentModerationService, ModerationRequest, Modality, ModerationResult
from src.preprocessor import TextPreprocessor, ImagePreprocessor, VideoPreprocessor, ContentPreprocessor
from src.model import RandomHateSpeechModel, RandomSexualModel, RandomViolenceModel, ContentModerationModel, ModelPrediction


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
    for field in required_fields:
        if field not in request_data:
            raise KeyError(field)
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


def format_success_response(predictions: dict[str, Union[ModelPrediction, list[ModelPrediction]]]) -> dict:
    """
    Format successful moderation response.

    Handles single predictions (text/image), video frames (list of predictions from one model),
    and multiple models predicting the same category.

    Args:
        predictions: Dictionary mapping category to prediction(s)

    Returns:
        Formatted success response dictionary
    """
    response = {"status": "success", "results": {}}
    for category, prediction in predictions.items():
        if isinstance(prediction, list):
            # Check if this is a video response (all predictions have same model_name)
            # or multiple models (different model_names)
            model_names = set(p.model_name for p in prediction)
            if len(model_names) == 1:
                # Single model with video frames
                frame_results = []
                for frame_idx, frame_prediction in enumerate(prediction):
                    scores = frame_prediction.to_dict()
                    avg_score = sum(scores.values()) / len(scores)
                    risk_level = RiskClassifier.classify_score(avg_score)
                    frame_results.append({
                        "frame": frame_idx,
                        "risk_level": risk_level.value,
                        "scores": scores
                    })
                response["results"][category] = {
                    "model_name": prediction[0].model_name,
                    "frames": frame_results
                }
            else:
                # Multiple models for the same category
                models_results = []
                for model_prediction in prediction:
                    scores = model_prediction.to_dict()
                    avg_score = sum(scores.values()) / len(scores)
                    risk_level = RiskClassifier.classify_score(avg_score)
                    models_results.append({
                        "model_name": model_prediction.model_name,
                        "risk_level": risk_level.value,
                        "scores": scores
                    })
                response["results"][category] = {
                    "models": models_results
                }
        else:
            # Text/image response (single prediction from single model)
            scores = prediction.to_dict()
            avg_score = sum(scores.values()) / len(scores)
            risk_level = RiskClassifier.classify_score(avg_score)
            response["results"][category] = {
                "model_name": prediction.model_name,
                "risk_level": risk_level.value,
                "scores": scores
            }
    return response


def format_error_response(error_message: str, status_code: int) -> dict:
    """
    Format error response.

    Args:
        error_message: Description of the error
        status_code: HTTP status code

    Returns:
        Dictionary ready to be JSON serialized
    """
    return {
        "status": "error",
        "error": error_message,
        "status_code": status_code
    }


class RequestHandler:
    """
    HTTP request handler for content moderation API.

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

    async def handle_moderate_request(self, request_body: str) -> dict:
        """
        Handle a content moderation request.

        Args:
            request_body: JSON string containing the request

        Returns:
            Dictionary with moderation result or error

        Raises:
            ValueError: If request format is invalid
            KeyError: If required fields are missing
        """
        try:
            request_data = json.loads(request_body)
            moderation_request: ModerationRequest = parse_request(request_data)
            result: ModerationResult = await self.service.moderate(moderation_request)
            return format_success_response(result.model_predictions)
        except ValueError as e:
            return format_error_response(str(e), 400)
        except KeyError as e:
            return format_error_response(f"Missing required field: {str(e)}", 400)
        except Exception as e:
            return format_error_response(f"Internal error: {str(e)}", 500)