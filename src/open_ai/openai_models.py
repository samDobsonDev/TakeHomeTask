from src.open_ai.openai_model_base import OpenAIContentModerationModel
from src.model import HateSpeechPrediction, SexualPrediction, ViolencePrediction


class OpenAIHateSpeechModel(OpenAIContentModerationModel[HateSpeechPrediction]):
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


class OpenAISexualModel(OpenAIContentModerationModel[SexualPrediction]):
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


class OpenAIViolenceModel(OpenAIContentModerationModel[ViolencePrediction]):
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