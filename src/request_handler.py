import asyncio
import base64
import json
from pathlib import Path
import requests
from src.content_loader import ContentLoader
from src.service import ContentModerationService, ModerationRequest, Modality, ModerationResult
from src.preprocessor import TextPreprocessor, ImagePreprocessor, VideoPreprocessor, ContentPreprocessor
from src.model import HateSpeechModel, SexualModel, ViolenceModel, ContentModerationModel


class ServiceContainer:
    """
    Dependency injection container for all service dependencies.

    Manages instantiation and lifecycle of preprocessors, models, and the service.
    """

    def __init__(self):
        """Initialize the container and all dependencies"""
        self._preprocessors = None
        self._models = None
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
                HateSpeechModel(),
                SexualModel(),
                ViolenceModel(),
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

    # Parse modality
    try:
        modality = Modality(request_data["modality"])
    except ValueError:
        raise ValueError(
            f"Invalid modality: {request_data['modality']}. Must be one of: {', '.join(m.value for m in Modality)}")

    # Validate content is not empty
    if not request_data["content"]:
        raise ValueError("Content cannot be empty")

    return ModerationRequest(
        content=request_data["content"],
        modality=modality,
        customer=request_data["customer"]
    )


def format_success_response(result: ModerationResult) -> dict:
    """
    Format successful moderation result as response.

    Args:
        result: ModerationResult from service

    Returns:
        Dictionary ready to be JSON serialized
    """
    return {
        "status": "success",
        "data": {
            category: {
                "risk_level": result.policy_classification.__getattribute__(category).value,
                "scores": prediction.to_dict()
            }
            for category, prediction in result.model_predictions.items()
        }
    }


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
            # Parse request
            request_data = json.loads(request_body)
            moderation_request: ModerationRequest = parse_request(request_data)
            # Process
            result: ModerationResult = await self.service.moderate(moderation_request)
            # Format response
            return format_success_response(result)

        except ValueError as e:
            return format_error_response(str(e), 400)
        except KeyError as e:
            return format_error_response(f"Missing required field: {str(e)}", 400)
        except Exception as e:
            return format_error_response(f"Internal error: {str(e)}", 500)


async def moderate_image_from_url():
    """Example 1: Download image from URL and moderate"""
    print("=" * 60)
    print("Example 1: Moderate Image from URL")
    print("=" * 60)
    image_url = "https://t3.ftcdn.net/jpg/03/21/62/56/360_F_321625657_rauGwvaYjtbETuwxn9kpBWKDYrVUMdB4.jpg"
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        base64_image = base64.b64encode(response.content).decode('utf-8')
        handler = RequestHandler()
        request_json = json.dumps({
            "content": base64_image,
            "modality": "image",
            "customer": "test_customer"
        })
        moderation_response = await handler.handle_moderate_request(request_json)
        print(json.dumps(moderation_response, indent=2))
    except requests.RequestException as e:
        print(f"Failed to download image: {e}")
    except Exception as e:
        print(f"Error: {e}")


async def moderate_image_from_local_file():
    """Example 2: Load image from local resources and moderate"""
    print("\n" + "=" * 60)
    print("Example 2: Moderate Image from Local File")
    print("=" * 60)
    try:
        # Load image using ContentLoader
        image_path = Path(__file__).parent / "resources" / "gun.png"
        image_bytes = ContentLoader.load_image(image_path)
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        handler = RequestHandler()
        request_json = json.dumps({
            "content": base64_image,
            "modality": "image",
            "customer": "test_customer"
        })
        moderation_response = await handler.handle_moderate_request(request_json)
        print(json.dumps(moderation_response, indent=2))
    except FileNotFoundError as e:
        print(f"Local image file not found: {e}")
    except Exception as e:
        print(f"Error: {e}")


async def main():
    # Example 1: From URL
    await moderate_image_from_url()
    # Example 2: From local file
    await moderate_image_from_local_file()


if __name__ == "__main__":
    asyncio.run(main())