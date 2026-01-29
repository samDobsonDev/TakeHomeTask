import pytest
from dataclasses import fields
from src.model import (
    HateSpeechPrediction,
    SexualPrediction,
    ViolencePrediction,
    ContentModerationModel,
    HateSpeechModel,
    SexualModel,
    ViolenceModel,
)
from src.preprocessor import PreprocessedText, PreprocessedImage, PreprocessedVideo


class TestModelPredictionDataclasses:
    """Test prediction dataclass structure and methods"""

    def test_hate_speech_prediction_has_correct_fields(self):
        """Verify HateSpeechPrediction has all required metric fields"""
        field_names = [f.name for f in fields(HateSpeechPrediction)]
        expected = ['input_data', 'toxicity', 'severe_toxicity', 'obscene',
                    'insult', 'identity_attack', 'threat']

        assert field_names == expected

    def test_sexual_prediction_has_correct_fields(self):
        """Verify SexualPrediction has all required metric fields"""
        field_names = [f.name for f in fields(SexualPrediction)]
        expected = ['input_data', 'sexual_explicit', 'adult_content', 'adult_toys']

        assert field_names == expected

    def test_violence_prediction_has_correct_fields(self):
        """Verify ViolencePrediction has all required metric fields"""
        field_names = [f.name for f in fields(ViolencePrediction)]
        expected = ['input_data', 'violence', 'firearm', 'knife']

        assert field_names == expected

    def test_hate_speech_prediction_get_category(self):
        """Verify HateSpeechPrediction returns correct category"""
        assert HateSpeechPrediction.get_category() == "hate_speech"

    def test_sexual_prediction_get_category(self):
        """Verify SexualPrediction returns correct category"""
        assert SexualPrediction.get_category() == "sexual"

    def test_violence_prediction_get_category(self):
        """Verify ViolencePrediction returns correct category"""
        assert ViolencePrediction.get_category() == "violence"


class TestModelPredictionToDict:
    """Test to_dict() method for predictions"""

    def test_hate_speech_prediction_to_dict(self):
        """Verify HateSpeechPrediction.to_dict() excludes input_data"""
        preprocessed = PreprocessedText(data=[1, 2, 3], original_text="test text")
        prediction = HateSpeechPrediction(
            input_data=preprocessed,
            toxicity=0.5,
            severe_toxicity=0.3,
            obscene=0.4,
            insult=0.6,
            identity_attack=0.5,
            threat=0.2
        )

        result = prediction.to_dict()

        assert 'input_data' not in result
        assert result['toxicity'] == 0.5
        assert result['severe_toxicity'] == 0.3
        assert len(result) == 6  # 6 metrics, no input_data

    def test_sexual_prediction_to_dict(self):
        """Verify SexualPrediction.to_dict() excludes input_data"""
        preprocessed = PreprocessedText(data=[1, 2, 3], original_text="test text")
        prediction = SexualPrediction(
            input_data=preprocessed,
            sexual_explicit=0.1,
            adult_content=0.2,
            adult_toys=0.3
        )
        result = prediction.to_dict()

        assert 'input_data' not in result
        assert result['sexual_explicit'] == 0.1
        assert len(result) == 3

    def test_violence_prediction_to_dict(self):
        """Verify ViolencePrediction.to_dict() excludes input_data"""
        preprocessed = PreprocessedText(data=[1, 2, 3], original_text="test text")
        prediction = ViolencePrediction(
            input_data=preprocessed,
            violence=0.7,
            firearm=0.8,
            knife=0.5
        )
        result = prediction.to_dict()

        assert 'input_data' not in result
        assert result['violence'] == 0.7
        assert len(result) == 3


class TestContentModerationModelValidation:
    """Test ContentModerationModel type validation"""

    def test_subclass_with_wrong_prediction_type_raises_error(self):
        """Verify that declaring wrong prediction type raises TypeError"""
        with pytest.raises(TypeError, match="declares.*but predict_text returns"):
            class BadModel(ContentModerationModel[HateSpeechPrediction]):  # noqa: F841
                async def predict_text(self, input_data: PreprocessedText) -> SexualPrediction:
                    pass

                async def predict_image(self, input_data: PreprocessedImage) -> HateSpeechPrediction:
                    pass

                async def predict_video(self, input_data: PreprocessedVideo) -> list[HateSpeechPrediction]:
                    pass


@pytest.mark.asyncio
class TestMockModelPredictions:
    """Test that mock models generate valid predictions"""

    async def test_hate_speech_model_predict_text_returns_valid_scores(self):
        """Verify HateSpeechModel.predict_text returns valid scores"""
        model = HateSpeechModel()
        preprocessed = PreprocessedText(data=[1] * 16, original_text="test text")
        prediction = await model.predict_text(preprocessed)

        assert isinstance(prediction, HateSpeechPrediction)
        assert 0 <= prediction.toxicity <= 1
        assert 0 <= prediction.severe_toxicity <= 1
        assert 0 <= prediction.obscene <= 1
        assert 0 <= prediction.insult <= 1
        assert 0 <= prediction.identity_attack <= 1
        assert 0 <= prediction.threat <= 1

    async def test_sexual_model_predict_image_returns_valid_scores(self):
        """Verify SexualModel.predict_image returns valid scores"""
        model = SexualModel()
        preprocessed = PreprocessedImage(data=[1] * 16, original_bytes=b"test image")
        prediction = await model.predict_image(preprocessed)

        assert isinstance(prediction, SexualPrediction)
        assert 0 <= prediction.sexual_explicit <= 1
        assert 0 <= prediction.adult_content <= 1
        assert 0 <= prediction.adult_toys <= 1

    async def test_violence_model_predict_image_returns_valid_scores(self):
        """Verify ViolenceModel.predict_image returns valid scores"""
        model = ViolenceModel()
        preprocessed = PreprocessedImage(data=[1] * 16, original_bytes=b"test image")#
        prediction = await model.predict_image(preprocessed)

        assert isinstance(prediction, ViolencePrediction)
        assert 0 <= prediction.violence <= 1
        assert 0 <= prediction.firearm <= 1
        assert 0 <= prediction.knife <= 1

    async def test_hate_speech_model_predict_video_returns_list_of_predictions(self):
        """Verify predict_video returns list of frame predictions"""
        model = HateSpeechModel()
        frame1 = PreprocessedImage(data=[1] * 16, original_bytes=b"frame1")
        frame2 = PreprocessedImage(data=[2] * 16, original_bytes=b"frame2")
        video = PreprocessedVideo(frames=[frame1, frame2])
        predictions = await model.predict_video(video)

        assert isinstance(predictions, list)
        assert len(predictions) == 2
        for prediction in predictions:
            assert isinstance(prediction, HateSpeechPrediction)
            assert 0 <= prediction.toxicity <= 1
            assert 0 <= prediction.severe_toxicity <= 1
            assert 0 <= prediction.obscene <= 1
            assert 0 <= prediction.insult <= 1
            assert 0 <= prediction.identity_attack <= 1
            assert 0 <= prediction.threat <= 1

    async def test_all_models_predict_video_returns_list_of_correct_type(self):
        """Verify all models return list of correct prediction type from predict_video"""
        frame1 = PreprocessedImage(data=[1] * 16, original_bytes=b"frame1")
        frame2 = PreprocessedImage(data=[2] * 16, original_bytes=b"frame2")
        video = PreprocessedVideo(frames=[frame1, frame2])
        hate_model = HateSpeechModel()
        hate_predictions = await hate_model.predict_video(video)

        assert isinstance(hate_predictions, list)
        assert len(hate_predictions) == 2
        assert all(isinstance(p, HateSpeechPrediction) for p in hate_predictions)

        sexual_model = SexualModel()
        sexual_predictions = await sexual_model.predict_video(video)

        assert isinstance(sexual_predictions, list)
        assert len(sexual_predictions) == 2
        assert all(isinstance(p, SexualPrediction) for p in sexual_predictions)

        violence_model = ViolenceModel()
        violence_predictions = await violence_model.predict_video(video)

        assert isinstance(violence_predictions, list)
        assert len(violence_predictions) == 2
        assert all(isinstance(p, ViolencePrediction) for p in violence_predictions)