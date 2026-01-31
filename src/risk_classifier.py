from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    """Risk levels for content moderation"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass
class PolicyClassification:
    """
    Final policy classification result after moderating content.
    
    Contains the risk level for each content category detected.
    Aggregates predictions from multiple models (if present) by taking the
    maximum risk level across all models for each category.
    
    Example:
        PolicyClassification(
            classifications={
                "violence": RiskLevel.HIGH,
                "hate_speech": RiskLevel.MEDIUM,
                "sexual": RiskLevel.LOW
            }
        )
    """
    classifications: dict[str, RiskLevel]


class RiskClassifier:
    """Classifies prediction scores into risk levels"""

    @staticmethod
    def classify_score(score: float) -> RiskLevel:
        """
        Classify a single averaged score into a risk level.

        Args:
            score: Average score (0.0-1.0)

        Returns:
            RiskLevel (LOW=1, MEDIUM=2, HIGH=3)
        """
        if score < 0.3:
            return RiskLevel.LOW
        elif score < 0.6:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.HIGH