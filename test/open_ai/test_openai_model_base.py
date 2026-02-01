import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pydantic import BaseModel
from src.open_ai.openai_model_base import OpenAIContentModerationModel
from src.model import (
    HateSpeechPrediction,
    SexualPrediction,
    ViolencePrediction,
    ModelPrediction,
)
from src.preprocessor import PreprocessedText, PreprocessedImage, PreprocessedVideo


# Concrete test implementation for HateSpeechPrediction
class TestHateSpeechModel(OpenAIContentModerationModel[HateSpeechPrediction]):
    """Test model for hate speech detection"""
    name = "TestHateSpeechModel"

    def get_model_name(self) -> str:
        return "gpt-4o-test"

    def get_text_prompt(self) -> str:
        return "Analyze for hate speech"

    def get_image_prompt(self) -> str:
        return "Analyze image for hate speech"


# Concrete test implementation for SexualPrediction
class TestSexualModel(OpenAIContentModerationModel[SexualPrediction]):
    """Test model for sexual content detection"""
    name = "TestSexualModel"

    def get_model_name(self) -> str:
        return "gpt-4o-test"

    def get_text_prompt(self) -> str:
        return "Analyze for sexual content"

    def get_image_prompt(self) -> str:
        return "Analyze image for sexual content"


# Concrete test implementation for ViolencePrediction
class TestViolenceModel(OpenAIContentModerationModel[ViolencePrediction]):
    """Test model for violence detection"""
    name = "TestViolenceModel"

    def get_model_name(self) -> str:
        return "gpt-4o-test"

    def get_text_prompt(self) -> str:
        return "Analyze for violence"

    def get_image_prompt(self) -> str:
        return "Analyze image for violence"


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client"""
    with patch('src.open_ai.openai_model_base.OpenAIClient') as mock:
        yield mock


@pytest.fixture
def hate_speech_model(mock_openai_client):
    """Create a TestHateSpeechModel with mocked client"""
    model = TestHateSpeechModel()
    model.client = MagicMock()
    return model


@pytest.fixture
def sexual_model(mock_openai_client):
    """Create a TestSexualModel with mocked client"""
    model = TestSexualModel()
    model.client = MagicMock()
    return model


@pytest.fixture
def violence_model(mock_openai_client):
    """Create a TestViolenceModel with mocked client"""
    model = TestViolenceModel()
    model.client = MagicMock()
    return model


class TestPredictionClassExtraction:
    """Tests for the _prediction_class property"""

    def test_extracts_hate_speech_prediction_class(self, hate_speech_model):
        """Should extract HateSpeechPrediction from generic type parameter"""
        assert hate_speech_model._prediction_class == HateSpeechPrediction

    def test_extracts_sexual_prediction_class(self, sexual_model):
        """Should extract SexualPrediction from generic type parameter"""
        assert sexual_model._prediction_class == SexualPrediction

    def test_extracts_violence_prediction_class(self, violence_model):
        """Should extract ViolencePrediction from generic type parameter"""
        assert violence_model._prediction_class == ViolencePrediction

    def test_prediction_class_is_model_prediction_subclass(self, hate_speech_model):
        """Extracted class should be a ModelPrediction subclass"""
        assert issubclass(hate_speech_model._prediction_class, ModelPrediction)


class TestGetResponseFormat:
    """Tests for the get_response_format method"""

    def test_returns_pydantic_base_model(self, hate_speech_model):
        """Should return a Pydantic BaseModel subclass"""
        response_format = hate_speech_model.get_response_format()
        assert issubclass(response_format, BaseModel)

    def test_hate_speech_response_format_has_correct_fields(self, hate_speech_model):
        """HateSpeech response format should have correct fields"""
        response_format = hate_speech_model.get_response_format()
        expected_fields = {
            "toxicity",
            "severe_toxicity",
            "obscene",
            "insult",
            "identity_attack",
            "threat",
        }
        assert set(response_format.model_fields.keys()) == expected_fields

    def test_sexual_response_format_has_correct_fields(self, sexual_model):
        """Sexual response format should have correct fields"""
        response_format = sexual_model.get_response_format()
        expected_fields = {"sexual_explicit", "adult_content", "adult_toys"}
        assert set(response_format.model_fields.keys()) == expected_fields

    def test_violence_response_format_has_correct_fields(self, violence_model):
        """Violence response format should have correct fields"""
        response_format = violence_model.get_response_format()
        expected_fields = {"violence", "firearm", "knife"}
        assert set(response_format.model_fields.keys()) == expected_fields

    def test_response_format_name_matches_prediction(self, hate_speech_model):
        """Response format name should be derived from prediction class"""
        response_format = hate_speech_model.get_response_format()
        assert response_format.__name__ == "HateSpeechScores"


class TestResponseToPrediction:
    """Tests for the _response_to_prediction method"""

    def test_converts_response_to_hate_speech_prediction(self, hate_speech_model):
        """Should convert Pydantic response to HateSpeechPrediction"""
        # Create a mock response with the expected fields
        response_format = hate_speech_model.get_response_format()
        mock_response = response_format(
            toxicity=0.8,
            severe_toxicity=0.3,
            obscene=0.5,
            insult=0.6,
            identity_attack=0.2,
            threat=0.1,
        )

        input_data = PreprocessedText(data=[1, 2, 3], original_text="test text")
        prediction = hate_speech_model._response_to_prediction(input_data, mock_response)

        assert isinstance(prediction, HateSpeechPrediction)
        assert prediction.toxicity == 0.8
        assert prediction.severe_toxicity == 0.3
        assert prediction.obscene == 0.5
        assert prediction.insult == 0.6
        assert prediction.identity_attack == 0.2
        assert prediction.threat == 0.1

    def test_sets_input_data_correctly(self, hate_speech_model):
        """Should set input_data on the prediction"""
        response_format = hate_speech_model.get_response_format()
        mock_response = response_format(
            toxicity=0.1, severe_toxicity=0.1, obscene=0.1,
            insult=0.1, identity_attack=0.1, threat=0.1,
        )

        input_data = PreprocessedText(data=[1, 2, 3], original_text="test text")
        prediction = hate_speech_model._response_to_prediction(input_data, mock_response)

        assert prediction.input_data == input_data

    def test_sets_model_name_correctly(self, hate_speech_model):
        """Should set model_name on the prediction"""
        response_format = hate_speech_model.get_response_format()
        mock_response = response_format(
            toxicity=0.1, severe_toxicity=0.1, obscene=0.1,
            insult=0.1, identity_attack=0.1, threat=0.1,
        )

        input_data = PreprocessedText(data=[1, 2, 3], original_text="test text")
        prediction = hate_speech_model._response_to_prediction(input_data, mock_response)

        assert prediction.model_name == "TestHateSpeechModel"

    def test_converts_sexual_response_to_prediction(self, sexual_model):
        """Should convert Pydantic response to SexualPrediction"""
        response_format = sexual_model.get_response_format()
        mock_response = response_format(
            sexual_explicit=0.7,
            adult_content=0.4,
            adult_toys=0.2,
        )

        input_data = PreprocessedText(data=[1, 2, 3], original_text="test text")
        prediction = sexual_model._response_to_prediction(input_data, mock_response)

        assert isinstance(prediction, SexualPrediction)
        assert prediction.sexual_explicit == 0.7
        assert prediction.adult_content == 0.4
        assert prediction.adult_toys == 0.2


class TestPredictText:
    """Tests for the predict_text method"""

    @pytest.mark.asyncio
    async def test_calls_client_analyze_text(self, hate_speech_model):
        """Should call client.analyze_text with correct parameters"""
        response_format = hate_speech_model.get_response_format()
        mock_response = response_format(
            toxicity=0.5, severe_toxicity=0.3, obscene=0.2,
            insult=0.4, identity_attack=0.1, threat=0.1,
        )
        hate_speech_model.client.analyze_text = AsyncMock(return_value=mock_response)

        input_data = PreprocessedText(data=[1, 2, 3], original_text="test content")
        await hate_speech_model.predict_text(input_data)

        hate_speech_model.client.analyze_text.assert_called_once_with(
            text="test content",
            model="gpt-4o-test",
            prompt="Analyze for hate speech",
            response_format=response_format,
        )

    @pytest.mark.asyncio
    async def test_returns_correct_prediction_type(self, hate_speech_model):
        """Should return HateSpeechPrediction instance"""
        response_format = hate_speech_model.get_response_format()
        mock_response = response_format(
            toxicity=0.5, severe_toxicity=0.3, obscene=0.2,
            insult=0.4, identity_attack=0.1, threat=0.1,
        )
        hate_speech_model.client.analyze_text = AsyncMock(return_value=mock_response)

        input_data = PreprocessedText(data=[1, 2, 3], original_text="test content")
        prediction = await hate_speech_model.predict_text(input_data)

        assert isinstance(prediction, HateSpeechPrediction)
        assert prediction.toxicity == 0.5


class TestPredictImage:
    """Tests for the predict_image method"""

    @pytest.mark.asyncio
    async def test_calls_client_analyze_image(self, hate_speech_model):
        """Should call client.analyze_image with correct parameters"""
        response_format = hate_speech_model.get_response_format()
        mock_response = response_format(
            toxicity=0.5, severe_toxicity=0.3, obscene=0.2,
            insult=0.4, identity_attack=0.1, threat=0.1,
        )
        hate_speech_model.client.analyze_image = AsyncMock(return_value=mock_response)

        input_data = PreprocessedImage(data=[1, 2, 3], original_bytes=b"image data")
        await hate_speech_model.predict_image(input_data)

        hate_speech_model.client.analyze_image.assert_called_once_with(
            image_bytes=b"image data",
            model="gpt-4o-test",
            prompt="Analyze image for hate speech",
            response_format=response_format,
        )

    @pytest.mark.asyncio
    async def test_returns_correct_prediction_type(self, hate_speech_model):
        """Should return HateSpeechPrediction instance"""
        response_format = hate_speech_model.get_response_format()
        mock_response = response_format(
            toxicity=0.7, severe_toxicity=0.5, obscene=0.3,
            insult=0.6, identity_attack=0.2, threat=0.4,
        )
        hate_speech_model.client.analyze_image = AsyncMock(return_value=mock_response)

        input_data = PreprocessedImage(data=[1, 2, 3], original_bytes=b"image data")
        prediction = await hate_speech_model.predict_image(input_data)

        assert isinstance(prediction, HateSpeechPrediction)
        assert prediction.toxicity == 0.7


class TestPredictVideo:
    """Tests for the predict_video method"""

    @pytest.mark.asyncio
    async def test_calls_client_analyze_video(self, hate_speech_model):
        """Should call client.analyze_video with correct parameters"""
        response_format = hate_speech_model.get_response_format()
        mock_responses = [
            response_format(
                toxicity=0.5, severe_toxicity=0.3, obscene=0.2,
                insult=0.4, identity_attack=0.1, threat=0.1,
            ),
            response_format(
                toxicity=0.6, severe_toxicity=0.4, obscene=0.3,
                insult=0.5, identity_attack=0.2, threat=0.2,
            ),
        ]
        hate_speech_model.client.analyze_video = AsyncMock(return_value=mock_responses)

        frame1 = PreprocessedImage(data=[1, 2], original_bytes=b"frame1")
        frame2 = PreprocessedImage(data=[3, 4], original_bytes=b"frame2")
        input_data = PreprocessedVideo(frames=[frame1, frame2])

        await hate_speech_model.predict_video(input_data)

        hate_speech_model.client.analyze_video.assert_called_once_with(
            frames=[b"frame1", b"frame2"],
            model="gpt-4o-test",
            prompt="Analyze image for hate speech",
            response_format=response_format,
        )

    @pytest.mark.asyncio
    async def test_returns_list_of_predictions(self, hate_speech_model):
        """Should return a list of predictions, one per frame"""
        response_format = hate_speech_model.get_response_format()
        mock_responses = [
            response_format(
                toxicity=0.5, severe_toxicity=0.3, obscene=0.2,
                insult=0.4, identity_attack=0.1, threat=0.1,
            ),
            response_format(
                toxicity=0.6, severe_toxicity=0.4, obscene=0.3,
                insult=0.5, identity_attack=0.2, threat=0.2,
            ),
        ]
        hate_speech_model.client.analyze_video = AsyncMock(return_value=mock_responses)

        frame1 = PreprocessedImage(data=[1, 2], original_bytes=b"frame1")
        frame2 = PreprocessedImage(data=[3, 4], original_bytes=b"frame2")
        input_data = PreprocessedVideo(frames=[frame1, frame2])

        predictions = await hate_speech_model.predict_video(input_data)

        assert len(predictions) == 2
        assert all(isinstance(p, HateSpeechPrediction) for p in predictions)
        assert predictions[0].toxicity == 0.5
        assert predictions[1].toxicity == 0.6

    @pytest.mark.asyncio
    async def test_each_prediction_has_correct_input_data(self, hate_speech_model):
        """Each prediction should reference its corresponding frame"""
        response_format = hate_speech_model.get_response_format()
        mock_responses = [
            response_format(
                toxicity=0.5, severe_toxicity=0.3, obscene=0.2,
                insult=0.4, identity_attack=0.1, threat=0.1,
            ),
            response_format(
                toxicity=0.6, severe_toxicity=0.4, obscene=0.3,
                insult=0.5, identity_attack=0.2, threat=0.2,
            ),
        ]
        hate_speech_model.client.analyze_video = AsyncMock(return_value=mock_responses)

        frame1 = PreprocessedImage(data=[1, 2], original_bytes=b"frame1")
        frame2 = PreprocessedImage(data=[3, 4], original_bytes=b"frame2")
        input_data = PreprocessedVideo(frames=[frame1, frame2])

        predictions = await hate_speech_model.predict_video(input_data)

        assert predictions[0].input_data == frame1
        assert predictions[1].input_data == frame2


class TestModelConfiguration:
    """Tests for model configuration methods"""

    def test_get_model_name_returns_configured_value(self, hate_speech_model):
        """get_model_name should return the configured OpenAI model"""
        assert hate_speech_model.get_model_name() == "gpt-4o-test"

    def test_get_text_prompt_returns_configured_value(self, hate_speech_model):
        """get_text_prompt should return the configured prompt"""
        assert hate_speech_model.get_text_prompt() == "Analyze for hate speech"

    def test_get_image_prompt_returns_configured_value(self, hate_speech_model):
        """get_image_prompt should return the configured prompt"""
        assert hate_speech_model.get_image_prompt() == "Analyze image for hate speech"

    def test_name_property_returns_model_name(self, hate_speech_model):
        """name property should return the model's name"""
        assert hate_speech_model.name == "TestHateSpeechModel"
