import pytest

from src.risk_classifier import RiskLevel, PolicyClassification, RiskClassifier


class TestRiskClassifierClassifyScore:
    """Test RiskClassifier.classify_score() method"""

    def test_classify_score_low_boundary_low(self):
        """Verify score < 0.3 is classified as LOW"""
        assert RiskClassifier.classify_score(0.0) == RiskLevel.LOW
        assert RiskClassifier.classify_score(0.1) == RiskLevel.LOW
        assert RiskClassifier.classify_score(0.29) == RiskLevel.LOW

    def test_classify_score_low_boundary_exact(self):
        """Verify score at 0.3 boundary transitions to MEDIUM"""
        assert RiskClassifier.classify_score(0.3) == RiskLevel.MEDIUM

    def test_classify_score_medium_range(self):
        """Verify 0.3 <= score < 0.6 is classified as MEDIUM"""
        assert RiskClassifier.classify_score(0.3) == RiskLevel.MEDIUM
        assert RiskClassifier.classify_score(0.5) == RiskLevel.MEDIUM
        assert RiskClassifier.classify_score(0.59) == RiskLevel.MEDIUM

    def test_classify_score_medium_boundary_exact(self):
        """Verify score at 0.6 boundary transitions to HIGH"""
        assert RiskClassifier.classify_score(0.6) == RiskLevel.HIGH

    def test_classify_score_high_range(self):
        """Verify score >= 0.6 is classified as HIGH"""
        assert RiskClassifier.classify_score(0.6) == RiskLevel.HIGH
        assert RiskClassifier.classify_score(0.8) == RiskLevel.HIGH
        assert RiskClassifier.classify_score(1.0) == RiskLevel.HIGH

    def test_classify_score_at_boundaries(self):
        """Verify exact boundary values"""
        assert RiskClassifier.classify_score(0.29999) == RiskLevel.LOW
        assert RiskClassifier.classify_score(0.30001) == RiskLevel.MEDIUM
        assert RiskClassifier.classify_score(0.59999) == RiskLevel.MEDIUM
        assert RiskClassifier.classify_score(0.60001) == RiskLevel.HIGH


class TestRiskClassifierAverageScores:
    """Test RiskClassifier.average_prediction_scores() method"""

    def test_average_scores_single_metric(self):
        """Verify averaging works with single metric"""
        scores = {"metric1": 0.5}
        assert RiskClassifier.average_prediction_scores(scores) == 0.5

    def test_average_scores_multiple_metrics(self):
        """Verify averaging works with multiple metrics"""
        scores = {
            "metric1": 0.2,
            "metric2": 0.4,
            "metric3": 0.6
        }
        assert RiskClassifier.average_prediction_scores(scores) == pytest.approx(0.4)

    def test_average_scores_all_same(self):
        """Verify averaging when all metrics are identical"""
        scores = {
            "metric1": 0.5,
            "metric2": 0.5,
            "metric3": 0.5,
            "metric4": 0.5
        }
        assert RiskClassifier.average_prediction_scores(scores) == 0.5

    def test_average_scores_zero(self):
        """Verify averaging with zero scores"""
        scores = {
            "metric1": 0.0,
            "metric2": 0.0
        }
        assert RiskClassifier.average_prediction_scores(scores) == 0.0

    def test_average_scores_one(self):
        """Verify averaging with maximum scores"""
        scores = {
            "metric1": 1.0,
            "metric2": 1.0
        }
        assert RiskClassifier.average_prediction_scores(scores) == 1.0

    def test_average_scores_empty_dict(self):
        """Verify averaging with empty dictionary returns 0.0"""
        scores = {}
        assert RiskClassifier.average_prediction_scores(scores) == 0.0

    def test_average_scores_mixed_values(self):
        """Verify averaging with mixed high and low values"""
        scores = {
            "low1": 0.1,
            "low2": 0.2,
            "high1": 0.9,
            "high2": 0.8
        }
        expected = (0.1 + 0.2 + 0.9 + 0.8) / 4
        assert RiskClassifier.average_prediction_scores(scores) == pytest.approx(expected)

    def test_average_scores_many_metrics(self):
        """Verify averaging with many metrics"""
        scores = {f"metric_{i}": i / 10.0 for i in range(11)}
        expected = sum(scores.values()) / len(scores)
        assert RiskClassifier.average_prediction_scores(scores) == pytest.approx(expected)


class TestRiskClassificationWorkflow:
    """Test complete classification workflow"""

    def test_classify_low_hate_speech(self):
        """Verify complete workflow for low hate speech scores"""
        scores = {
            "toxicity": 0.1,
            "severe_toxicity": 0.2,
            "obscene": 0.15
        }

        avg = RiskClassifier.average_prediction_scores(scores)
        risk_level = RiskClassifier.classify_score(avg)

        assert risk_level == RiskLevel.LOW

    def test_classify_medium_sexual_content(self):
        """Verify complete workflow for medium sexual content scores"""
        scores = {
            "sexual_explicit": 0.4,
            "adult_content": 0.5,
            "adult_toys": 0.45
        }

        avg = RiskClassifier.average_prediction_scores(scores)
        risk_level = RiskClassifier.classify_score(avg)

        assert risk_level == RiskLevel.MEDIUM

    def test_classify_high_violence(self):
        """Verify complete workflow for high violence scores"""
        scores = {
            "violence": 0.8,
            "firearm": 0.9,
            "knife": 0.85
        }

        avg = RiskClassifier.average_prediction_scores(scores)
        risk_level = RiskClassifier.classify_score(avg)

        assert risk_level == RiskLevel.HIGH

    def test_policy_classification_from_workflow(self):
        """Verify creating PolicyClassification from classification workflow"""
        hate_scores = {"toxicity": 0.1, "insult": 0.2}
        sexual_scores = {"sexual_explicit": 0.5}
        violence_scores = {"violence": 0.8, "firearm": 0.9}

        hate_avg = RiskClassifier.average_prediction_scores(hate_scores)
        sexual_avg = RiskClassifier.average_prediction_scores(sexual_scores)
        violence_avg = RiskClassifier.average_prediction_scores(violence_scores)

        classification = PolicyClassification(
            hate_speech=RiskClassifier.classify_score(hate_avg),
            sexual=RiskClassifier.classify_score(sexual_avg),
            violence=RiskClassifier.classify_score(violence_avg)
        )

        assert classification.hate_speech == RiskLevel.LOW
        assert classification.sexual == RiskLevel.MEDIUM
        assert classification.violence == RiskLevel.HIGH