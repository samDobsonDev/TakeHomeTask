import pytest
from pydantic import BaseModel, ValidationError
from src.pydantic_generator import prediction_to_pydantic
from src.model import (
    HateSpeechPrediction,
    SexualPrediction,
    ViolencePrediction,
)


class TestPredictionToPydantic:
    """Tests for the prediction_to_pydantic function"""

    def test_returns_pydantic_base_model_subclass(self):
        """Generated class should be a Pydantic BaseModel subclass"""
        result = prediction_to_pydantic(HateSpeechPrediction)
        assert issubclass(result, BaseModel)

    def test_model_name_replaces_prediction_with_scores(self):
        """Model name should replace 'Prediction' with 'Scores'"""
        result = prediction_to_pydantic(HateSpeechPrediction)
        assert result.__name__ == "HateSpeechScores"

        result = prediction_to_pydantic(SexualPrediction)
        assert result.__name__ == "SexualScores"

        result = prediction_to_pydantic(ViolencePrediction)
        assert result.__name__ == "ViolenceScores"

    def test_excludes_input_data_field(self):
        """Generated model should not include input_data field"""
        result = prediction_to_pydantic(HateSpeechPrediction)
        assert "input_data" not in result.model_fields

    def test_excludes_model_name_field(self):
        """Generated model should not include model_name field"""
        result = prediction_to_pydantic(HateSpeechPrediction)
        assert "model_name" not in result.model_fields

    def test_hate_speech_prediction_fields(self):
        """HateSpeechPrediction should generate model with correct fields"""
        result = prediction_to_pydantic(HateSpeechPrediction)
        expected_fields = {
            "toxicity",
            "severe_toxicity",
            "obscene",
            "insult",
            "identity_attack",
            "threat",
        }
        assert set(result.model_fields.keys()) == expected_fields

    def test_sexual_prediction_fields(self):
        """SexualPrediction should generate model with correct fields"""
        result = prediction_to_pydantic(SexualPrediction)
        expected_fields = {"sexual_explicit", "adult_content", "adult_toys"}
        assert set(result.model_fields.keys()) == expected_fields

    def test_violence_prediction_fields(self):
        """ViolencePrediction should generate model with correct fields"""
        result = prediction_to_pydantic(ViolencePrediction)
        expected_fields = {"violence", "firearm", "knife"}
        assert set(result.model_fields.keys()) == expected_fields

    def test_all_fields_are_float_type(self):
        """All generated fields should be float type"""
        result = prediction_to_pydantic(HateSpeechPrediction)
        for field_name, field_info in result.model_fields.items():
            assert field_info.annotation == float, f"Field {field_name} should be float"

    def test_all_fields_are_required(self):
        """All generated fields should be required (no defaults)"""
        result = prediction_to_pydantic(HateSpeechPrediction)
        for field_name, field_info in result.model_fields.items():
            assert field_info.is_required(), f"Field {field_name} should be required"


class TestCachingBehavior:
    """Tests for the caching behavior of prediction_to_pydantic"""

    def test_same_class_returns_same_model(self):
        """Calling with same prediction class should return identical model"""
        result1 = prediction_to_pydantic(HateSpeechPrediction)
        result2 = prediction_to_pydantic(HateSpeechPrediction)
        assert result1 is result2

    def test_different_classes_return_different_models(self):
        """Different prediction classes should return different models"""
        hate_speech_model = prediction_to_pydantic(HateSpeechPrediction)
        sexual_model = prediction_to_pydantic(SexualPrediction)
        violence_model = prediction_to_pydantic(ViolencePrediction)

        assert hate_speech_model is not sexual_model
        assert hate_speech_model is not violence_model
        assert sexual_model is not violence_model


class TestGeneratedModelInstantiation:
    """Tests for instantiating generated Pydantic models"""

    def test_can_instantiate_with_valid_data(self):
        """Generated model should accept valid float data"""
        scores_cls = prediction_to_pydantic(HateSpeechPrediction)
        instance = scores_cls(
            toxicity=0.5,
            severe_toxicity=0.3,
            obscene=0.1,
            insult=0.2,
            identity_attack=0.4,
            threat=0.6,
        )
        assert instance.toxicity == 0.5
        assert instance.severe_toxicity == 0.3
        assert instance.obscene == 0.1
        assert instance.insult == 0.2
        assert instance.identity_attack == 0.4
        assert instance.threat == 0.6

    def test_can_instantiate_sexual_scores(self):
        """SexualScores model should accept valid data"""
        scores_cls = prediction_to_pydantic(SexualPrediction)
        instance = scores_cls(
            sexual_explicit=0.1,
            adult_content=0.2,
            adult_toys=0.3,
        )
        assert instance.sexual_explicit == 0.1
        assert instance.adult_content == 0.2
        assert instance.adult_toys == 0.3

    def test_can_instantiate_violence_scores(self):
        """ViolenceScores model should accept valid data"""
        scores_cls = prediction_to_pydantic(ViolencePrediction)
        instance = scores_cls(
            violence=0.7,
            firearm=0.8,
            knife=0.9,
        )
        assert instance.violence == 0.7
        assert instance.firearm == 0.8
        assert instance.knife == 0.9


class TestGeneratedModelValidation:
    """Tests for Pydantic validation on generated models"""

    def test_rejects_missing_required_field(self):
        """Should raise ValidationError when required field is missing"""
        scores_cls = prediction_to_pydantic(HateSpeechPrediction)
        with pytest.raises(ValidationError) as exc_info:
            scores_cls(
                toxicity=0.5,
                # Missing other required fields
            )
        # Should mention missing fields
        assert "severe_toxicity" in str(exc_info.value)

    def test_coerces_int_to_float(self):
        """Pydantic should coerce int values to float"""
        scores_cls = prediction_to_pydantic(ViolencePrediction)
        instance = scores_cls(violence=1, firearm=0, knife=1)
        assert instance.violence == 1.0
        assert isinstance(instance.violence, float)

    def test_rejects_non_numeric_value(self):
        """Should raise ValidationError for non-numeric values"""
        scores_cls = prediction_to_pydantic(HateSpeechPrediction)
        with pytest.raises(ValidationError):
            scores_cls(
                toxicity="not a number",
                severe_toxicity=0.3,
                obscene=0.1,
                insult=0.2,
                identity_attack=0.4,
                threat=0.6,
            )

    def test_model_fields_accessible_via_model_fields(self):
        """model_fields should be accessible for introspection"""
        scores_cls = prediction_to_pydantic(HateSpeechPrediction)
        assert hasattr(scores_cls, "model_fields")
        assert len(scores_cls.model_fields.keys()) == 6


class TestModelSerialization:
    """Tests for serialization of generated models"""

    def test_model_dump_returns_dict(self):
        """model_dump() should return a dictionary of field values"""
        scores_cls = prediction_to_pydantic(HateSpeechPrediction)
        instance = scores_cls(
            toxicity=0.5,
            severe_toxicity=0.3,
            obscene=0.1,
            insult=0.2,
            identity_attack=0.4,
            threat=0.6,
        )
        result = instance.model_dump()
        assert result == {
            "toxicity": 0.5,
            "severe_toxicity": 0.3,
            "obscene": 0.1,
            "insult": 0.2,
            "identity_attack": 0.4,
            "threat": 0.6,
        }

    def test_model_dump_json_returns_json_string(self):
        """model_dump_json() should return a JSON string"""
        scores_cls = prediction_to_pydantic(ViolencePrediction)
        instance = scores_cls(violence=0.7, firearm=0.8, knife=0.9)
        result = instance.model_dump_json()
        assert isinstance(result, str)
        assert '"violence":0.7' in result or '"violence": 0.7' in result
