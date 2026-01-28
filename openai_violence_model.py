import base64
from dataclasses import fields
from openai import OpenAI
from pydantic import BaseModel
from model import ContentModerationModel, ViolencePrediction
from preprocessor import PreprocessedImage, PreprocessedText


class ViolenceScores(BaseModel):
    """Structured output for violence scores"""
    violence: float
    firearm: float
    knife: float


class OpenAIViolenceModel(ContentModerationModel[ViolencePrediction]):
    """
    Violence detection model using OpenAI's vision API with structured outputs.
    """

    def __init__(self, api_key: str = None):
        """Initialize the OpenAI violence model."""
        self.client = OpenAI(api_key=api_key)

    async def predict_text(self, input_data: PreprocessedText) -> ViolencePrediction:
        """Violence detection on text is not supported"""
        raise NotImplementedError("OpenAI vision model requires image/video input")

    async def predict_image(self, input_data: PreprocessedImage) -> ViolencePrediction:
        """Detect violence in image using OpenAI Responses API with structured outputs"""
        image_bytes = input_data.original_bytes
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        response = self.client.responses.parse(
            model="gpt-4o-2024-08-06",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Analyze this image for violent content. Provide scores between 0 and 1 for: violence (overall level), firearm (presence), knife (presence)."
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    ],
                }
            ],
            text_format=ViolenceScores,
        )
        # Get parsed output
        scores_obj = response.output_parsed
        # Extract scores dynamically from prediction class fields
        metric_names = [f.name for f in fields(ViolencePrediction) if f.name != 'input_data']
        scores = {
            metric: max(0.0, min(1.0, getattr(scores_obj, metric)))
            for metric in metric_names
        }
        return ViolencePrediction(input_data=input_data, **scores)