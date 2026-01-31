import pytest
import json
import base64
from unittest.mock import MagicMock
from src.service import Modality, ModerationRequest, ModerationResult
from src.risk_classifier import RiskLevel, PolicyClassification
from src.request_handler import (
    ServiceContainer,
    RequestHandler,
    parse_request,
    format_success_response,
    format_error_response,
)


def build_moderation_result(predictions_dict: dict, policy_classifications: dict[str, RiskLevel] = None) -> ModerationResult:
    """
    Helper to build a ModerationResult for testing format_success_response.
    
    Args:
        predictions_dict: Dict mapping category to prediction(s). Single predictions are auto-wrapped in lists.
        policy_classifications: Optional dict mapping category to RiskLevel (computed if not provided)
    
    Returns:
        ModerationResult with PolicyClassification and model_predictions
    """
    # Normalize predictions_dict: wrap single predictions in lists
    normalized_predictions: dict = {}
    for category, prediction in predictions_dict.items():
        if isinstance(prediction, list):
            normalized_predictions[category] = prediction
        else:
            # Wrap single prediction in a list
            normalized_predictions[category] = [prediction]
    if policy_classifications is None:
        # Auto-compute risk levels from predictions
        policy_classifications = {}
        for category, predictions_list in normalized_predictions.items():
            from src.score_calculator import ScoreCalculator
            from src.risk_classifier import RiskClassifier
            # Compute max score across all predictions for this category
            scores_list = [ScoreCalculator.compute_average_score(p) for p in predictions_list]
            max_score = max(scores_list)
            policy_classifications[category] = RiskClassifier.classify_score(max_score)
    
    policy_classification = PolicyClassification(classifications=policy_classifications)
    return ModerationResult(
        policy_classification=policy_classification,
        model_predictions=normalized_predictions
    )


class TestServiceContainer:
    """Test ServiceContainer dependency injection"""

    def test_container_lazy_loads_preprocessors(self):
        """Verify preprocessors are lazy-loaded"""
        container = ServiceContainer()
        assert container._preprocessors is None
        preprocessors = container.preprocessors
        assert preprocessors is not None
        assert "text" in preprocessors
        assert "image" in preprocessors
        assert "video" in preprocessors

    def test_container_lazy_loads_models(self):
        """Verify models are lazy-loaded"""
        container = ServiceContainer()
        assert container._models is None
        models = container.models
        assert models is not None
        assert len(models) == 3

    def test_container_lazy_loads_service(self):
        """Verify service is lazy-loaded"""
        container = ServiceContainer()
        assert container._service is None

        service = container.service

        assert service is not None

    def test_container_reuses_instances(self):
        """Verify container reuses instances"""
        container = ServiceContainer()
        preprocessors1 = container.preprocessors
        preprocessors2 = container.preprocessors

        assert preprocessors1 is preprocessors2


class TestParseRequest:
    """Test parse_request helper function"""

    def test_parse_valid_text_request(self):
        """Verify parsing valid text request"""
        request_data = {
            "content": "test content",
            "modality": "text",
            "customer": "test_customer"
        }
        result = parse_request(request_data)

        assert isinstance(result, ModerationRequest)
        assert result.content == "test content"
        assert result.modality == Modality.TEXT
        assert result.customer == "test_customer"

    def test_parse_valid_image_request(self):
        """Verify parsing valid image request"""
        request_data = {
            "content": base64.b64encode(b"image").decode('utf-8'),
            "modality": "image",
            "customer": "test_customer"
        }
        result = parse_request(request_data)

        assert result.modality == Modality.IMAGE

    def test_parse_valid_video_request(self):
        """Verify parsing valid video request with list of frames"""
        frame1_b64 = base64.b64encode(b"frame1").decode('utf-8')
        frame2_b64 = base64.b64encode(b"frame2").decode('utf-8')
        request_data = {
            "content": [frame1_b64, frame2_b64],
            "modality": "video",
            "customer": "test_customer"
        }
        result = parse_request(request_data)

        assert result.modality == Modality.VIDEO
        assert isinstance(result.content, list)
        assert len(result.content) == 2

    def test_parse_missing_content_field(self):
        """Verify missing content raises KeyError"""
        request_data = {
            "modality": "text",
            "customer": "test"
        }

        with pytest.raises(KeyError):
            parse_request(request_data)

    def test_parse_missing_modality_field(self):
        """Verify missing modality raises KeyError"""
        request_data = {
            "content": "test",
            "customer": "test"
        }

        with pytest.raises(KeyError):
            parse_request(request_data)

    def test_parse_missing_customer_field(self):
        """Verify missing customer raises KeyError"""
        request_data = {
            "content": "test",
            "modality": "text"
        }

        with pytest.raises(KeyError):
            parse_request(request_data)

    def test_parse_invalid_modality(self):
        """Verify invalid modality raises ValueError"""
        request_data = {
            "content": "test",
            "modality": "invalid_modality",
            "customer": "test"
        }

        with pytest.raises(ValueError, match="Invalid modality"):
            parse_request(request_data)

    def test_parse_empty_content(self):
        """Verify empty content raises ValueError"""
        request_data = {
            "content": "",
            "modality": "text",
            "customer": "test"
        }

        with pytest.raises(ValueError, match="Content cannot be empty"):
            parse_request(request_data)

    def test_parse_empty_video_content(self):
        """Verify empty video content raises ValueError"""
        request_data = {
            "content": [],
            "modality": "video",
            "customer": "test"
        }

        with pytest.raises(ValueError, match="Video content cannot be empty"):
            parse_request(request_data)

    def test_parse_video_content_not_list(self):
        """Verify non-list video content raises ValueError"""
        request_data = {
            "content": "single_frame_b64",
            "modality": "video",
            "customer": "test"
        }

        with pytest.raises(ValueError, match="Video content must be a list"):
            parse_request(request_data)


class TestFormatSuccessResponse:
    """Test format_success_response helper function"""

    def test_format_success_response_single_prediction_structure(self):
        """Verify success response has correct structure for single prediction"""
        mock_prediction: MagicMock = MagicMock()
        mock_prediction.to_dict.return_value = {"metric1": 0.5}
        mock_prediction.model_name = "TestModel"
        # For single model per category, only include one category
        predictions_dict: dict = {
            "hate_speech": mock_prediction
        }
        moderation_result: ModerationResult = build_moderation_result(predictions_dict)
        response = format_success_response(moderation_result)

        assert response.status == "success"
        assert "hate_speech" in response.results

    def test_format_success_response_includes_risk_levels(self):
        """Verify success response includes risk levels"""
        mock_prediction: MagicMock = MagicMock()
        mock_prediction.to_dict.return_value = {"metric": 0.5}
        mock_prediction.model_name = "TestModel"
        # Test with multiple categories to verify each gets a risk level
        predictions_dict: dict = {
            "hate_speech": mock_prediction,
            "sexual": mock_prediction,
            "violence": mock_prediction
        }
        moderation_result: ModerationResult = build_moderation_result(predictions_dict)
        response = format_success_response(moderation_result)

        assert response.results["hate_speech"].risk_level is not None
        assert isinstance(response.results["hate_speech"].risk_level, RiskLevel)
        assert response.results["sexual"].risk_level is not None
        assert response.results["violence"].risk_level is not None

    def test_format_success_response_includes_scores(self):
        """Verify success response includes detailed scores"""
        mock_prediction: MagicMock = MagicMock()
        mock_prediction.to_dict.return_value = {"metric1": 0.5, "metric2": 0.3}
        mock_prediction.model_name = "TestModel"
        # For single model per category, predictions_dict should have one model per category
        predictions_dict: dict = {
            "hate_speech": mock_prediction
        }
        moderation_result: ModerationResult = build_moderation_result(predictions_dict)
        response = format_success_response(moderation_result)

        assert response.results["hate_speech"].single_model is not None
        assert response.results["hate_speech"].single_model.scores["metric1"] == 0.5
        assert response.results["hate_speech"].single_model.scores["metric2"] == 0.3

    def test_format_success_response_includes_model_name(self):
        """Verify success response includes model_name"""
        mock_prediction: MagicMock = MagicMock()
        mock_prediction.to_dict.return_value = {"metric": 0.5}
        mock_prediction.model_name = "HateSpeechModel"
        predictions_dict: dict = {
            "hate_speech": mock_prediction
        }
        moderation_result: ModerationResult = build_moderation_result(predictions_dict)
        response = format_success_response(moderation_result)

        assert response.results["hate_speech"].single_model.model_name == "HateSpeechModel"

    def test_format_success_response_video_predictions(self):
        """Verify success response for video predictions with frames"""
        mock_prediction1: MagicMock = MagicMock()
        mock_prediction1.to_dict.return_value = {"metric": 0.3}
        mock_prediction1.model_name = "TestModel"
        mock_prediction2: MagicMock = MagicMock()
        mock_prediction2.to_dict.return_value = {"metric": 0.7}
        mock_prediction2.model_name = "TestModel"
        # For video: single model with multiple frames, wrap in another list
        # model_predictions structure: {category: [model1_frames_list]} where model1_frames_list = [frame1, frame2]
        predictions_dict: dict = {
            "hate_speech": [[mock_prediction1, mock_prediction2]]
        }
        moderation_result: ModerationResult = build_moderation_result(predictions_dict)
        response = format_success_response(moderation_result)

        assert response.status == "success"
        assert response.results["hate_speech"].single_model is not None
        assert len(response.results["hate_speech"].single_model.frames) == 2
        assert response.results["hate_speech"].single_model.frames[0].frame == 0
        assert response.results["hate_speech"].single_model.frames[1].frame == 1

    def test_format_success_response_video_includes_model_name(self):
        """Verify success response for video predictions includes model_name at category level"""
        mock_prediction1: MagicMock = MagicMock()
        mock_prediction1.to_dict.return_value = {"metric": 0.3}
        mock_prediction1.model_name = "ViolenceModel"
        mock_prediction2: MagicMock = MagicMock()
        mock_prediction2.to_dict.return_value = {"metric": 0.7}
        mock_prediction2.model_name = "ViolenceModel"
        # For video: single model with multiple frames, wrap in another list
        predictions_dict: dict = {
            "violence": [[mock_prediction1, mock_prediction2]]
        }
        moderation_result: ModerationResult = build_moderation_result(predictions_dict)
        response = format_success_response(moderation_result)

        # Model name at model result level
        assert response.results["violence"].single_model.model_name == "ViolenceModel"
        # Should have frames (single model video frames)
        assert len(response.results["violence"].single_model.frames) == 2
        assert response.results["violence"].single_model.frames[0].frame == 0
        assert response.results["violence"].single_model.frames[1].frame == 1

    def test_format_success_response_multiple_models_same_category(self):
        """Verify success response for multiple models predicting same category"""
        mock_prediction1: MagicMock = MagicMock()
        mock_prediction1.to_dict.return_value = {"metric": 0.3}
        mock_prediction1.model_name = "RandomViolenceModel"
        mock_prediction2: MagicMock = MagicMock()
        mock_prediction2.to_dict.return_value = {"metric": 0.7}
        mock_prediction2.model_name = "OpenAIViolenceModel"
        predictions_dict: dict = {
            "violence": [mock_prediction1, mock_prediction2]
        }
        moderation_result: ModerationResult = build_moderation_result(predictions_dict)
        response = format_success_response(moderation_result)

        # Should have models array (multiple models)
        assert len(response.results["violence"].models) == 2
        assert response.results["violence"].models[0].model_name == "RandomViolenceModel"
        assert response.results["violence"].models[1].model_name == "OpenAIViolenceModel"
        # Should NOT have single_model
        assert response.results["violence"].single_model is None


class TestFormatErrorResponse:
    """Test format_error_response helper function"""

    def test_format_error_response_structure(self):
        """Verify error response has correct structure"""
        response = format_error_response("Test error", 400)

        assert response.status == "error"
        assert response.error == "Test error"
        assert response.status_code == 400

    def test_format_error_response_400(self):
        """Verify 400 error response"""
        response = format_error_response("Bad request", 400)

        assert response.status_code == 400

    def test_format_error_response_500(self):
        """Verify 500 error response"""
        response = format_error_response("Internal error", 500)

        assert response.status_code == 500


class TestRequestHandler:
    """Test RequestHandler"""

    def test_request_handler_initialization(self):
        """Verify RequestHandler initializes with container"""
        handler = RequestHandler()

        assert handler.container is not None
        assert handler.service is not None

    def test_request_handler_with_custom_container(self):
        """Verify RequestHandler accepts custom container"""
        custom_container = ServiceContainer()
        handler = RequestHandler(custom_container)

        assert handler.container is custom_container

    @pytest.mark.asyncio
    async def test_handle_moderate_request_valid_text(self):
        """Verify handling valid text moderation request"""
        handler = RequestHandler()
        request_json = json.dumps({
            "content": "test content",
            "modality": "text",
            "customer": "test_customer"
        })
        response = await handler.handle_moderate_request(request_json)

        assert response.status == "success"
        assert len(response.results) > 0

    @pytest.mark.asyncio
    async def test_handle_moderate_request_valid_video(self):
        """Verify handling valid video moderation request"""
        handler = RequestHandler()
        frame1_b64 = base64.b64encode(b"frame1").decode('utf-8')
        frame2_b64 = base64.b64encode(b"frame2").decode('utf-8')
        request_json = json.dumps({
            "content": [frame1_b64, frame2_b64],
            "modality": "video",
            "customer": "test_customer"
        })
        response = await handler.handle_moderate_request(request_json)

        assert response.status == "success"
        assert len(response.results) > 0

    @pytest.mark.asyncio
    async def test_handle_moderate_request_invalid_json(self):
        """Verify handling invalid JSON request"""
        handler = RequestHandler()
        response = await handler.handle_moderate_request("invalid json")

        assert response.status == "error"
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_handle_moderate_request_missing_field(self):
        """Verify handling request with missing field"""
        handler = RequestHandler()
        request_json = json.dumps({
            "content": "test",
            "modality": "text"
            # Missing customer field
        })
        response = await handler.handle_moderate_request(request_json)

        assert response.status == "error"
        assert response.status_code == 400
        assert "Missing required field" in response.error

    @pytest.mark.asyncio
    async def test_handle_moderate_request_invalid_modality(self):
        """Verify handling request with invalid modality"""
        handler = RequestHandler()
        request_json = json.dumps({
            "content": "test",
            "modality": "invalid",
            "customer": "test"
        })
        response = await handler.handle_moderate_request(request_json)

        assert response.status == "error"
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_handle_moderate_request_empty_content(self):
        """Verify handling request with empty content"""
        handler = RequestHandler()
        request_json = json.dumps({
            "content": "",
            "modality": "text",
            "customer": "test"
        })
        response = await handler.handle_moderate_request(request_json)

        assert response.status == "error"
        assert response.status_code == 400
        assert "empty" in response.error.lower()

    @pytest.mark.asyncio
    async def test_handle_moderate_request_video_not_list(self):
        """Verify handling video request with non-list content"""
        handler = RequestHandler()
        request_json = json.dumps({
            "content": "single_frame_b64",
            "modality": "video",
            "customer": "test"
        })
        response = await handler.handle_moderate_request(request_json)

        assert response.status == "error"
        assert response.status_code == 400