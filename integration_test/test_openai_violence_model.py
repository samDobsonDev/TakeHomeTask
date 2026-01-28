import os
from typing import cast

import pytest
import requests
import base64
from dotenv import load_dotenv

from model import ViolencePrediction
from openai_violence_model import OpenAIViolenceModel
from preprocessor import PreprocessedText, PreprocessedImage, TextPreprocessor, ImagePreprocessor, VideoPreprocessor
from service import ContentModerationService, Modality, ModerationRequest
from risk_classifier import RiskLevel

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

@pytest.mark.asyncio
async def test_violence_model_rejects_text_input():
    """Test that violence model raises NotImplementedError for text input"""
    model = OpenAIViolenceModel(api_key=OPENAI_API_KEY)
    preprocessed_text = PreprocessedText(data=[1, 2, 3])
    with pytest.raises(NotImplementedError, match="OpenAI vision model requires image/video input"):
        await model.predict_text(preprocessed_text)

@pytest.mark.asyncio
async def test_violence_model_detects_weapon_in_image():
    """Test violence detection on image with weapon (gun)"""
    model = OpenAIViolenceModel(api_key=OPENAI_API_KEY)
    image_url = "https://t3.ftcdn.net/jpg/03/21/62/56/360_F_321625657_rauGwvaYjtbETuwxn9kpBWKDYrVUMdB4.jpg"
    response = requests.get(image_url)
    response.raise_for_status()
    image_bytes = response.content
    preprocessed_image = PreprocessedImage(data=[1] * 16, original_bytes=image_bytes)
    prediction = await model.predict_image(preprocessed_image)
    # Assert prediction is valid
    assert prediction is not None
    assert hasattr(prediction, 'violence')
    assert hasattr(prediction, 'firearm')
    assert hasattr(prediction, 'knife')
    # Assert scores are in valid range
    assert 0 <= prediction.violence <= 1
    assert 0 <= prediction.firearm <= 1
    assert 0 <= prediction.knife <= 1
    # Weapon image should have higher firearm score
    assert prediction.firearm > 0.3, f"Expected firearm score > 0.3, got {prediction.firearm}"

@pytest.mark.asyncio
async def test_violence_model_with_harmless_image():
    """Test violence detection on harmless image (computer desk)"""
    model = OpenAIViolenceModel(api_key=OPENAI_API_KEY)
    # Using a simple harmless image URL
    image_url = "https://images.unsplash.com/photo-1587825140708-dfaf72ae4b04?w=400"
    response = requests.get(image_url)
    response.raise_for_status()
    image_bytes = response.content
    preprocessed_image = PreprocessedImage(data=[1] * 16, original_bytes=image_bytes)
    prediction = await model.predict_image(preprocessed_image)
    # Assert prediction is valid
    assert prediction is not None
    assert 0 <= prediction.violence <= 1
    assert 0 <= prediction.firearm <= 1
    assert 0 <= prediction.knife <= 1
    # Harmless image should have lower violence scores
    assert prediction.firearm < 0.5, f"Expected firearm score < 0.5 for harmless image, got {prediction.firearm}"

@pytest.mark.asyncio
async def test_full_pipeline_with_openai_model_weapon_image():
    """Integration test: Full moderation pipeline with OpenAI model on weapon image"""
    preprocessors = {
        "image": ImagePreprocessor(),
    }
    models = [OpenAIViolenceModel(api_key=OPENAI_API_KEY)]
    service = ContentModerationService(preprocessors=preprocessors, models=models)

    # Download weapon image
    image_url = "https://t3.ftcdn.net/jpg/03/21/62/56/360_F_321625657_rauGwvaYjtbETuwxn9kpBWKDYrVUMdB4.jpg"
    response = requests.get(image_url)
    response.raise_for_status()
    image_bytes = response.content
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    # Create moderation request
    request = ModerationRequest(
        content=base64_image,
        modality=Modality.IMAGE,
        customer="integration_test"
    )

    # Run full pipeline
    result = await service.moderate(request)

    # Verify result structure
    assert result is not None
    assert result.policy_classification is not None
    assert result.model_predictions is not None

    # Verify violence predictions exist
    assert "violence" in result.model_predictions
    violence_prediction = cast(ViolencePrediction, result.model_predictions["violence"])

    # Verify prediction has valid scores
    assert violence_prediction.violence >= 0
    assert violence_prediction.firearm >= 0
    assert violence_prediction.knife >= 0

    # Verify risk classification - weapon image should result in HIGH violence risk
    assert result.policy_classification.violence == RiskLevel.HIGH


@pytest.mark.asyncio
async def test_full_pipeline_with_openai_model_harmless_image():
    """Integration test: Full moderation pipeline with OpenAI model on harmless image"""
    # Setup service with OpenAI violence model
    preprocessors = {
        "text": TextPreprocessor(),
        "image": ImagePreprocessor(),
        "video": VideoPreprocessor()
    }

    models = [OpenAIViolenceModel(api_key=OPENAI_API_KEY)]

    service = ContentModerationService(preprocessors=preprocessors, models=models)

    # Download harmless image
    image_url = "https://images.unsplash.com/photo-1587825140708-dfaf72ae4b04?w=400"
    response = requests.get(image_url)
    response.raise_for_status()
    image_bytes = response.content
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    # Create moderation request
    request = ModerationRequest(
        content=base64_image,
        modality=Modality.IMAGE,
        customer="integration_test"
    )

    # Run full pipeline
    result = await service.moderate(request)

    # Verify result structure
    assert result is not None
    assert result.policy_classification is not None
    assert result.model_predictions is not None

    # Verify violence predictions exist
    assert "violence" in result.model_predictions
    violence_prediction = cast(ViolencePrediction, result.model_predictions["violence"])

    # Verify prediction has valid scores
    assert 0 <= violence_prediction.violence <= 1
    assert 0 <= violence_prediction.firearm <= 1
    assert 0 <= violence_prediction.knife <= 1

    # Verify risk classification - non-weapon image should result in LOW violence risk
    result.policy_classification.violence = RiskLevel.LOW

    # Harmless image should result in low violence risk
    print(f"Violence risk level: {result.policy_classification.violence.value}")
    print(f"Firearm score: {violence_prediction.firearm}")