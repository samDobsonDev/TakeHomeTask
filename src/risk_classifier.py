from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    """Risk levels for content moderation"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class PolicyClassification:
    """Final policy classification result"""
    hate_speech: RiskLevel
    sexual: RiskLevel
    violence: RiskLevel

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary format"""
        return {
            "hate_speech": self.hate_speech.value,
            "sexual": self.sexual.value,
            "violence": self.violence.value
        }


class RiskClassifier:
    """Classifies prediction scores into risk levels"""

    @staticmethod
    def classify_score(score: float) -> RiskLevel:
        """
        Classify a single averaged score into a risk level.

        Args:
            score: Average score (0.0-1.0)

        Returns:
            RiskLevel (low, medium, or high)
        """
        if score < 0.3:
            return RiskLevel.LOW
        elif score < 0.6:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.HIGH

    @staticmethod
    def average_prediction_scores(scores: dict[str, float]) -> float:
        """
        Average all scores from a model prediction.

        Args:
            scores: Dictionary of metric scores

        Returns:
            Average score across all metrics
        """
        if not scores:
            return 0.0
        return sum(scores.values()) / len(scores)