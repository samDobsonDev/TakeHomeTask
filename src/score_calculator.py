from typing import Union
from src.model import ModelPrediction


class ScoreCalculator:
    """Calculates average scores from model predictions."""

    @staticmethod
    def compute_average_score(prediction: Union[ModelPrediction, list[ModelPrediction]]) -> float:
        """
        Compute average score from a prediction (single or list of frames).
        
        Args:
            prediction: Either a single ModelPrediction or a list of ModelPredictions (video frames)
        
        Returns:
            Average score across all values
        """
        if isinstance(prediction, list):
            # Video: flatten all scores across all frames
            all_scores = [v for frame in prediction for v in frame.to_dict().values()]
        else:
            # Text/image: get scores from single prediction
            all_scores = list(prediction.to_dict().values())
        return sum(all_scores) / len(all_scores) if all_scores else 0.0
