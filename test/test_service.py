import base64
import pytest
from service import (
    Modality,
    PredictionType,
    ModerationRequest,
    ModerationResult,
    ContentModerationService,
)
from preprocessor import PreprocessedText, PreprocessedImage, PreprocessedVideo, TextPreprocessor, ImagePreprocessor, \
    VideoPreprocessor
from model import HateSpeechModel, SexualModel, ViolenceModel
from risk_classifier import RiskLevel, PolicyClassification


class TestModality:
    """Test Modality enum"""

    def test_modality_text_value(self):
        """Verify TEXT modality has correct value"""
        assert Modality.TEXT.value == "text"

    def test_modality_image_value(self):
        """Verify IMAGE modality has correct value"""
        assert Modality.IMAGE.value == "image"

    def test_modality_video_value(self):
        """Verify VIDEO modality has correct value"""
        assert Modality.VIDEO.value == "video"


class TestPredictionType:
    """Test PredictionType enum"""

    def test_prediction_type_policy_value(self):
        """Verify POLICY prediction type has correct value"""
        assert PredictionType.POLICY.value == "policy"


class TestModerationRequest:
    """Test ModerationRequest dataclass"""

    def test_moderation_request_creation(self):
        """Verify ModerationRequest can be created"""
        request = ModerationRequest(
            content="test content",
            modality=Modality.TEXT,
            customer="test_customer"
        )
        assert request.content == "test content"
        assert request.modality == Modality.TEXT
        assert request.customer == "test_customer"
        assert request.prediction_type == PredictionType.POLICY

    def test_moderation_request_default_prediction_type(self):
        """Verify default prediction_type is POLICY"""
        request = ModerationRequest(
            content="content",
            modality=Modality.IMAGE,
            customer="customer"
        )
        assert request.prediction_type == PredictionType.POLICY


class TestModerationResult:
    """Test ModerationResult dataclass"""

    def test_moderation_result_creation(self):
        classification = PolicyClassification(
            hate_speech=RiskLevel.LOW,
            sexual=RiskLevel.MEDIUM,
            violence=RiskLevel.HIGH
        )
        predictions = {}
        result = ModerationResult(
            policy_classification=classification,
            model_predictions=predictions
        )
        assert result.policy_classification == classification
        assert result.model_predictions == predictions


class TestContentModerationService:
    """Test ContentModerationService"""

    @pytest.fixture
    def service(self):
        """Create a service with mock preprocessors and models"""
        preprocessors = {
            "text": TextPreprocessor(),
            "image": ImagePreprocessor(),
            "video": VideoPreprocessor()
        }
        models = [
            HateSpeechModel(),
            SexualModel(),
            ViolenceModel()
        ]
        return ContentModerationService(preprocessors=preprocessors, models=models)

    @pytest.mark.asyncio
    async def test_moderate_text_content(self, service):
        """Verify moderating text content"""
        request = ModerationRequest(
            content="This is a test text",
            modality=Modality.TEXT,
            customer="test_customer"
        )
        result = await service.moderate(request)
        assert isinstance(result, ModerationResult)
        assert result.policy_classification is not None
        assert len(result.model_predictions) == 3  # 3 models

    @pytest.mark.asyncio
    async def test_moderate_image_content(self, service):
        """Verify moderating image content"""
        image_bytes = b"fake_image_bytes"
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        request = ModerationRequest(
            content=base64_image,
            modality=Modality.IMAGE,
            customer="test_customer"
        )
        result = await service.moderate(request)
        assert isinstance(result, ModerationResult)
        assert result.policy_classification is not None
        assert len(result.model_predictions) == 3

    @pytest.mark.asyncio
    async def test_moderate_video_content(self, service):
        """Verify moderating video content"""
        # Video is list of bytes (frames) that gets base64 encoded
        frame_bytes = b"frame1" + b"frame2"
        base64_video = base64.b64encode(frame_bytes).decode('utf-8')
        request = ModerationRequest(
            content=base64_video,
            modality=Modality.VIDEO,
            customer="test_customer"
        )
        result = await service.moderate(request)
        assert isinstance(result, ModerationResult)
        assert result.policy_classification is not None
        assert len(result.model_predictions) == 3

    @pytest.mark.asyncio
    async def test_policy_classification_has_all_categories(self, service):
        """Verify policy classification includes all 3 categories"""
        request = ModerationRequest(
            content="test",
            modality=Modality.TEXT,
            customer="test"
        )
        result = await service.moderate(request)
        classification = result.policy_classification
        assert hasattr(classification, 'hate_speech')
        assert hasattr(classification, 'sexual')
        assert hasattr(classification, 'violence')
        assert isinstance(classification.hate_speech, RiskLevel)
        assert isinstance(classification.sexual, RiskLevel)
        assert isinstance(classification.violence, RiskLevel)

    @pytest.mark.asyncio
    async def test_model_predictions_have_correct_categories(self, service):
        """Verify model predictions are keyed by category"""
        request = ModerationRequest(
            content="test",
            modality=Modality.TEXT,
            customer="test"
        )
        result = await service.moderate(request)
        expected_categories = {"hate_speech", "sexual", "violence"}
        assert set(result.model_predictions.keys()) == expected_categories

    @pytest.mark.asyncio
    async def test_predictions_contain_scores(self, service):
        """Verify predictions have scores that can be converted to dict"""
        request = ModerationRequest(
            content="test",
            modality=Modality.TEXT,
            customer="test"
        )
        result = await service.moderate(request)
        for category, prediction in result.model_predictions.items():
            scores_dict = prediction.to_dict()
            assert isinstance(scores_dict, dict)
            assert len(scores_dict) > 0
            # All scores should be floats between 0-1
            assert all(isinstance(v, float) and 0 <= v <= 1 for v in scores_dict.values())

    @pytest.mark.asyncio
    async def test_moderate_with_valid_request(self, service):
        """Verify service successfully moderates valid requests"""
        request = ModerationRequest(
            content="Valid content to moderate",
            modality=Modality.TEXT,
            customer="test_customer"
        )
        result = await service.moderate(request)
        assert result is not None
        assert result.policy_classification is not None

    @pytest.mark.asyncio
    async def test_moderation_with_different_customers(self, service):
        """Verify service works with different customers"""
        request1 = ModerationRequest(
            content="test",
            modality=Modality.TEXT,
            customer="customer1"
        )
        request2 = ModerationRequest(
            content="test",
            modality=Modality.TEXT,
            customer="customer2"
        )
        result1 = await service.moderate(request1)
        result2 = await service.moderate(request2)
        assert result1.policy_classification is not None
        assert result2.policy_classification is not None


class TestContentModerationServiceIntegration:
    """Integration tests for ContentModerationService"""

    @pytest.mark.asyncio
    async def test_full_moderation_pipeline_text(self):
        """Test complete moderation pipeline for text"""
        preprocessors = {
            "text": TextPreprocessor(),
            "image": ImagePreprocessor(),
            "video": VideoPreprocessor()
        }
        models = [HateSpeechModel(), SexualModel(), ViolenceModel()]
        service = ContentModerationService(preprocessors, models)
        request = ModerationRequest(
            content="Test content for moderation",
            modality=Modality.TEXT,
            customer="integration_test"
        )
        result = await service.moderate(request)
        # Verify complete result structure
        assert result.policy_classification.hate_speech in RiskLevel
        assert result.policy_classification.sexual in RiskLevel
        assert result.policy_classification.violence in RiskLevel
        assert len(result.model_predictions) == 3

    @pytest.mark.asyncio
    async def test_risk_levels_are_valid(self):
        """Verify all risk levels are valid RiskLevel enum values"""
        preprocessors = {
            "text": TextPreprocessor(),
            "image": ImagePreprocessor(),
            "video": VideoPreprocessor()
        }
        models = [HateSpeechModel(), SexualModel(), ViolenceModel()]
        service = ContentModerationService(preprocessors, models)
        request = ModerationRequest(
            content="test",
            modality=Modality.TEXT,
            customer="test"
        )
        result = await service.moderate(request)
        classification = result.policy_classification
        valid_levels = {RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH}
        assert classification.hate_speech in valid_levels
        assert classification.sexual in valid_levels
        assert classification.violence in valid_levels