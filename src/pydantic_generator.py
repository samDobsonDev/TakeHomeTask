from dataclasses import fields
from functools import lru_cache
from typing import Type
from pydantic import create_model, BaseModel
from src.model import ModelPrediction


@lru_cache(maxsize=None)
def prediction_to_pydantic(prediction_class: Type[ModelPrediction]) -> Type[BaseModel]:
    """
    Dynamically generate a Pydantic model from a ModelPrediction dataclass.

    Extracts all float score fields (excluding input_data and model_name) and creates
    a Pydantic model with those fields. This ensures the Pydantic model always stays
    in sync with the prediction dataclass.

    The result is cached, so repeated calls with the same prediction class
    return the same Pydantic model class.

    Args:
        prediction_class: A ModelPrediction subclass (e.g., HateSpeechPrediction)

    Returns:
        A Pydantic BaseModel class with the same score fields
    """
    # Get all fields except the base class fields
    score_fields = {
        f.name: (float, ...)  # (type, default) - ... means required
        for f in fields(prediction_class)
        if f.name not in ('input_data', 'model_name')
    }
    # Generate model name from prediction class name
    model_name = prediction_class.__name__.replace('Prediction', 'Scores')
    # Create the Pydantic model dynamically
    return create_model(model_name, **score_fields)
