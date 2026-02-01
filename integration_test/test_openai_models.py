import os
import pytest
import requests
import base64
from dotenv import load_dotenv
from src.model import ViolencePrediction
from src.open_ai.openai_models import OpenAIHateSpeechModel, OpenAISexualModel, OpenAIViolenceModel
from src.preprocessor import PreprocessedText, PreprocessedImage, TextPreprocessor, ImagePreprocessor, PreprocessedVideo
from src.service import ContentModerationService, Modality, ModerationRequest
from src.risk_classifier import RiskLevel

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

def fetch_image_from_url(url: str) -> bytes:
    """Fetch image from URL and return as bytes"""
    response = requests.get(url)
    response.raise_for_status()  # Raise exception for bad status codes
    return response.content

class TestOpenAIHateSpeechModel:
    """Test OpenAI hate speech model"""

    @pytest.mark.asyncio
    async def test_predict_text_returns_valid_prediction(self):
        """Test hate speech prediction on text"""
        model = OpenAIHateSpeechModel(api_key=OPENAI_API_KEY)
        preprocessed_text = PreprocessedText(original_text="test", data=[1] * 16)
        prediction = await model.predict_text(preprocessed_text)

        assert prediction is not None
        assert hasattr(prediction, 'toxicity')
        assert hasattr(prediction, 'severe_toxicity')
        assert 0 <= prediction.toxicity <= 1
        assert 0 <= prediction.severe_toxicity <= 1

    @pytest.mark.asyncio
    async def test_predict_image_returns_valid_prediction(self):
        """Test hate speech prediction on image"""
        model = OpenAIHateSpeechModel(api_key=OPENAI_API_KEY)
        image_bytes = fetch_image_from_url(
            "https://i.etsystatic.com/11012956/r/il/9b5fb7/3870999220/il_570xN.3870999220_6vfu.jpg"
        )
        preprocessed_image = PreprocessedImage(original_bytes=image_bytes, data=[1] * 16)
        prediction = await model.predict_image(preprocessed_image)

        assert prediction is not None
        assert all(0 <= getattr(prediction, field) <= 1 for field in [
            'toxicity', 'severe_toxicity', 'obscene', 'insult', 'identity_attack', 'threat'
        ])


class TestOpenAISexualModel:
    """Test OpenAI sexual content model"""

    @pytest.mark.asyncio
    async def test_predict_text_returns_valid_prediction(self):
        """Test sexual content prediction on text"""
        model = OpenAISexualModel(api_key=OPENAI_API_KEY)
        preprocessed_text = PreprocessedText(original_text="test", data=[1] * 16)
        prediction = await model.predict_text(preprocessed_text)

        assert prediction is not None
        assert hasattr(prediction, 'sexual_explicit')
        assert hasattr(prediction, 'adult_content')
        assert 0 <= prediction.sexual_explicit <= 1
        assert 0 <= prediction.adult_content <= 1

    @pytest.mark.asyncio
    async def test_predict_image_returns_valid_prediction(self):
        """Test sexual content prediction on image"""
        model = OpenAISexualModel(api_key=OPENAI_API_KEY)
        image_bytes = fetch_image_from_url(
            "https://media.post.rvohealth.io/wp-content/uploads/sites/3/2020/02/321428_2200-800x1200.jpg"
        )
        preprocessed_image = PreprocessedImage(original_bytes=image_bytes, data=[1] * 16)
        prediction = await model.predict_image(preprocessed_image)

        assert prediction is not None
        assert all(0 <= getattr(prediction, field) <= 1
                   for field in ['sexual_explicit', 'adult_content', 'adult_toys'])


class TestOpenAIViolenceModel:
    """Test OpenAI violence detection model"""

    @pytest.mark.asyncio
    async def test_predict_text_returns_valid_prediction(self):
        """Test violence prediction on text"""
        model = OpenAIViolenceModel(api_key=OPENAI_API_KEY)
        preprocessed_text = PreprocessedText(original_text="test", data=[1] * 16)
        prediction = await model.predict_text(preprocessed_text)

        assert prediction is not None
        assert hasattr(prediction, 'violence')
        assert hasattr(prediction, 'firearm')
        assert 0 <= prediction.violence <= 1
        assert 0 <= prediction.firearm <= 1

    @pytest.mark.asyncio
    async def test_predict_image_detects_weapon(self):
        """Test violence detection on weapon image"""
        model = OpenAIViolenceModel(api_key=OPENAI_API_KEY)
        image_bytes = fetch_image_from_url(
            "https://t3.ftcdn.net/jpg/03/21/62/56/360_F_321625657_rauGwvaYjtbETuwxn9kpBWKDYrVUMdB4.jpg"
        )
        preprocessed_image = PreprocessedImage(original_bytes=image_bytes, data=[1] * 16)
        prediction = await model.predict_image(preprocessed_image)

        assert prediction is not None
        assert prediction.firearm > 0.3, f"Expected firearm score > 0.3, got {prediction.firearm}"

    @pytest.mark.asyncio
    async def test_predict_image_safe_content(self):
        """Test violence detection on harmless image"""
        model = OpenAIViolenceModel(api_key=OPENAI_API_KEY)
        image_bytes = fetch_image_from_url(
            "https://images.unsplash.com/photo-1587825140708-dfaf72ae4b04?w=400"
        )
        preprocessed_image = PreprocessedImage(original_bytes=image_bytes, data=[1] * 16)
        prediction = await model.predict_image(preprocessed_image)

        assert prediction is not None
        assert prediction.firearm < 0.5, f"Expected firearm score < 0.5, got {prediction.firearm}"

    @pytest.mark.asyncio
    async def test_analyze_video_with_5_frames(self):
        """Test violence prediction on video with 5 frames"""
        model = OpenAIViolenceModel(api_key=OPENAI_API_KEY)
        gun_image_bytes = fetch_image_from_url(
            "https://t3.ftcdn.net/jpg/03/21/62/56/360_F_321625657_rauGwvaYjtbETuwxn9kpBWKDYrVUMdB4.jpg"
        )
        frames = [
            PreprocessedImage(
                original_bytes=gun_image_bytes,
                data=[1] * 16
            )
            for _ in range(5)
        ]
        video = PreprocessedVideo(frames=frames)
        predictions = await model.predict_video(video)

        assert predictions is not None
        assert isinstance(predictions, list)
        assert len(predictions) == 5
        for prediction in predictions:
            assert isinstance(prediction, ViolencePrediction)
            assert all(0 <= getattr(prediction, field) <= 1 for field in [
                'violence', 'firearm', 'knife'
            ])
            # At least one violence metric should be non-zero for gun image
            assert any(getattr(prediction, field) > 0 for field in [
                'violence', 'firearm', 'knife'
            ])


class TestFullPipelineWithOpenAIModels:
    """Integration tests for full moderation pipeline with OpenAI models"""

    @pytest.mark.asyncio
    async def test_pipeline_with_hate_speech_model(self):
        """Integration test: full pipeline with hate speech model"""
        preprocessors = {
            "text": TextPreprocessor()
        }
        models = [OpenAIHateSpeechModel(api_key=OPENAI_API_KEY)]
        service = ContentModerationService(preprocessors=preprocessors, models=models)
        request = ModerationRequest(
            content="test content",
            modality=Modality.TEXT,
            customer="integration_test"
        )
        result = await service.moderate(request)

        assert result is not None
        assert result.policy_classification is not None
        assert "hate_speech" in result.model_predictions

    @pytest.mark.asyncio
    async def test_pipeline_with_violence_model_weapon_image(self):
        """Integration test: violence model on weapon image"""
        preprocessors = {
            "image": ImagePreprocessor()
        }
        models = [OpenAIViolenceModel(api_key=OPENAI_API_KEY)]
        service = ContentModerationService(preprocessors=preprocessors, models=models)
        image = fetch_image_from_url("https://t3.ftcdn.net/jpg/03/21/62/56/360_F_321625657_rauGwvaYjtbETuwxn9kpBWKDYrVUMdB4.jpg")
        base64_image = base64.b64encode(image).decode('utf-8')
        request = ModerationRequest(
            content=base64_image,
            modality=Modality.IMAGE,
            customer="integration_test"
        )
        result = await service.moderate(request)

        assert result is not None
        assert "violence" in result.policy_classification.classifications
        assert result.policy_classification.classifications["violence"] in {RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH}