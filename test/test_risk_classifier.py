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


class TestRiskClassificationWorkflow:
    """Test complete classification workflow"""

    def test_classify_low_hate_speech(self):
        """Verify complete workflow for low hate speech scores"""
        scores = {
            "toxicity": 0.1,
            "severe_toxicity": 0.2,
            "obscene": 0.15
        }
        avg = sum(scores.values()) / len(scores)
        risk_level = RiskClassifier.classify_score(avg)

        assert risk_level == RiskLevel.LOW

    def test_classify_medium_sexual_content(self):
        """Verify complete workflow for medium sexual content scores"""
        scores = {
            "sexual_explicit": 0.4,
            "adult_content": 0.5,
            "adult_toys": 0.45
        }
        avg = sum(scores.values()) / len(scores)
        risk_level = RiskClassifier.classify_score(avg)

        assert risk_level == RiskLevel.MEDIUM

    def test_classify_high_violence(self):
        """Verify complete workflow for high violence scores"""
        scores = {
            "violence": 0.8,
            "firearm": 0.9,
            "knife": 0.85
        }
        avg = sum(scores.values()) / len(scores)
        risk_level = RiskClassifier.classify_score(avg)

        assert risk_level == RiskLevel.HIGH

    def test_policy_classification_from_workflow(self):
        """Verify creating PolicyClassification from classification workflow"""
        hate_scores = {"toxicity": 0.1, "insult": 0.2}
        sexual_scores = {"sexual_explicit": 0.5}
        violence_scores = {"violence": 0.8, "firearm": 0.9}
        hate_avg = sum(hate_scores.values()) / len(hate_scores)
        sexual_avg = sum(sexual_scores.values()) / len(sexual_scores)
        violence_avg = sum(violence_scores.values()) / len(violence_scores)
        classification = PolicyClassification(
            classifications={
                "hate_speech": RiskClassifier.classify_score(hate_avg),
                "sexual": RiskClassifier.classify_score(sexual_avg),
                "violence": RiskClassifier.classify_score(violence_avg)
            }
        )

        assert classification.classifications["hate_speech"] == RiskLevel.LOW
        assert classification.classifications["sexual"] == RiskLevel.MEDIUM
        assert classification.classifications["violence"] == RiskLevel.HIGH

    def test_policy_classification_partial_categories(self):
        """Verify PolicyClassification works with only some categories"""
        hate_scores = {"toxicity": 0.1, "insult": 0.2}
        hate_avg = sum(hate_scores.values()) / len(hate_scores)
        classification = PolicyClassification(
            classifications={
                "hate_speech": RiskClassifier.classify_score(hate_avg)
            }
        )

        assert "hate_speech" in classification.classifications
        assert "sexual" not in classification.classifications
        assert "violence" not in classification.classifications
        assert classification.classifications["hate_speech"] == RiskLevel.LOW