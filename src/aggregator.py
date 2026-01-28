from dataclasses import dataclass


@dataclass
class ScoreAggregator:
    """Aggregates multiple prediction scores by averaging"""

    @staticmethod
    def average_scores(predictions: list[dict[str, float]]) -> dict[str, float]:
        """
        Average scores across multiple predictions.

        Args:
            predictions: List of score dictionaries from multiple predictions

        Returns:
            Dictionary with averaged scores
        """
        if not predictions:
            return {}
        aggregated = {}
        metric_names = predictions[0].keys()
        for metric in metric_names:
            avg_score = sum(prediction[metric] for prediction in predictions) / len(predictions)
            aggregated[metric] = avg_score
        return aggregated