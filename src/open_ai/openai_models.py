from src.open_ai.openai_model_base import OpenAIContentModerationModel
from src.model import HateSpeechPrediction, SexualPrediction, ViolencePrediction
from src.preprocessor import PreprocessedContent
from pydantic import BaseModel


class HateSpeechScores(BaseModel):
    """Structured output for hate speech scores"""
    toxicity: float
    severe_toxicity: float
    obscene: float
    insult: float
    identity_attack: float
    threat: float


class SexualScores(BaseModel):
    """Structured output for sexual content scores"""
    sexual_explicit: float
    adult_content: float
    adult_toys: float


class ViolenceScores(BaseModel):
    """Structured output for violence scores"""
    violence: float
    firearm: float
    knife: float


class OpenAIHateSpeechModel(OpenAIContentModerationModel[HateSpeechPrediction, HateSpeechScores]):
    """OpenAI-powered hate speech detection model"""
    name = "OpenAIHateSpeechModel"

    def get_model_name(self) -> str:
        return "gpt-4o-2024-08-06"

    def get_text_prompt(self) -> str:
        return """Analyze this text for hate speech content. 
        Return scores between 0.0 and 1.0 for each metric, where:
        - 0.0 = no detection
        - 1.0 = maximum severity

        Metrics: toxicity, severe_toxicity, obscene, insult, identity_attack, threat"""

    def get_image_prompt(self) -> str:
        return """Analyze this image for hate speech content. 
        Return scores between 0.0 and 1.0 for each metric, where:
        - 0.0 = no detection
        - 1.0 = maximum severity

        Metrics: toxicity, severe_toxicity, obscene, insult, identity_attack, threat"""

    def get_response_format(self) -> type[HateSpeechScores]:
        return HateSpeechScores

    def _response_to_prediction(self, input_data: PreprocessedContent,
                                response: HateSpeechScores) -> HateSpeechPrediction:
        """Convert response to prediction"""
        scores = {field_name: getattr(response, field_name) for field_name in response.model_fields.keys()}
        return HateSpeechPrediction(input_data=input_data, model_name=self.name, **scores)


class OpenAISexualModel(OpenAIContentModerationModel[SexualPrediction, SexualScores]):
    """OpenAI-powered sexual content detection model"""
    name = "OpenAISexualModel"

    def get_model_name(self) -> str:
        return "gpt-4o-2024-08-06"

    def get_text_prompt(self) -> str:
        return """Analyze this text for sexual content. 
        Return scores between 0.0 and 1.0 for each metric, where:
        - 0.0 = no detection
        - 1.0 = maximum severity

        Metrics: sexual_explicit, adult_content, adult_toys"""

    def get_image_prompt(self) -> str:
        return """Analyze this image for sexual content. 
        Return scores between 0.0 and 1.0 for each metric, where:
        - 0.0 = no detection
        - 1.0 = maximum severity

        Metrics: sexual_explicit, adult_content, adult_toys"""

    def get_response_format(self) -> type[SexualScores]:
        return SexualScores

    def _response_to_prediction(self, input_data: PreprocessedContent,
                                response: SexualScores) -> SexualPrediction:
        """Convert response to prediction"""
        scores = {field_name: getattr(response, field_name) for field_name in response.model_fields.keys()}
        return SexualPrediction(input_data=input_data, model_name=self.name, **scores)


class OpenAIViolenceModel(OpenAIContentModerationModel[ViolencePrediction, ViolenceScores]):
    """OpenAI-powered violence detection model"""
    name = "OpenAIViolenceModel"

    def get_model_name(self) -> str:
        return "gpt-4o-2024-08-06"

    def get_text_prompt(self) -> str:
        return """Analyze this text for violence content. 
        Return scores between 0.0 and 1.0 for each metric, where:
        - 0.0 = no detection
        - 1.0 = maximum severity

        Metrics: violence, firearm, knife"""

    def get_image_prompt(self) -> str:
        return """Analyze this image for violence content. 
        Return scores between 0.0 and 1.0 for each metric, where:
        - 0.0 = no detection
        - 1.0 = maximum severity

        Metrics: violence, firearm, knife"""

    def get_response_format(self) -> type[ViolenceScores]:
        return ViolenceScores

    def _response_to_prediction(self, input_data: PreprocessedContent,
                                response: ViolenceScores) -> ViolencePrediction:
        """Convert response to prediction"""
        scores = {field_name: getattr(response, field_name) for field_name in response.model_fields.keys()}
        return ViolencePrediction(input_data=input_data, model_name=self.name, **scores)