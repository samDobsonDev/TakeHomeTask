import pytest
from aggregator import ScoreAggregator


class TestScoreAggregator:
    """Test ScoreAggregator.average_scores() method"""

    def test_average_scores_single_prediction(self):
        """Verify averaging works with single prediction"""
        predictions = [
            {"toxicity": 0.5, "insult": 0.3, "threat": 0.2}
        ]

        result = ScoreAggregator.average_scores(predictions)

        assert result["toxicity"] == 0.5
        assert result["insult"] == 0.3
        assert result["threat"] == 0.2

    def test_average_scores_multiple_predictions(self):
        """Verify averaging across multiple predictions"""
        predictions = [
            {"violence": 0.2, "firearm": 0.4},
            {"violence": 0.4, "firearm": 0.6},
            {"violence": 0.6, "firearm": 0.8}
        ]

        result = ScoreAggregator.average_scores(predictions)

        assert result["violence"] == pytest.approx(0.4)
        assert result["firearm"] == pytest.approx(0.6)

    def test_average_scores_empty_list(self):
        """Verify averaging with empty predictions list returns empty dict"""
        predictions = []

        result = ScoreAggregator.average_scores(predictions)

        assert result == {}

    def test_average_scores_all_same_values(self):
        """Verify averaging when all predictions are identical"""
        predictions = [
            {"metric1": 0.5, "metric2": 0.5},
            {"metric1": 0.5, "metric2": 0.5},
            {"metric1": 0.5, "metric2": 0.5}
        ]

        result = ScoreAggregator.average_scores(predictions)

        assert result["metric1"] == 0.5
        assert result["metric2"] == 0.5

    def test_average_scores_extreme_values(self):
        """Verify averaging with extreme values (0 and 1)"""
        predictions = [
            {"risky": 0.0},
            {"risky": 1.0}
        ]

        result = ScoreAggregator.average_scores(predictions)

        assert result["risky"] == pytest.approx(0.5)

    def test_average_scores_many_predictions(self):
        """Verify averaging across many predictions"""
        predictions = [
            {"score": float(i) / 10.0} for i in range(11)  # 0.0, 0.1, ..., 1.0
        ]

        result = ScoreAggregator.average_scores(predictions)

        expected_avg = sum(i / 10.0 for i in range(11)) / 11
        assert result["score"] == pytest.approx(expected_avg)

    def test_average_scores_many_metrics(self):
        """Verify averaging with many different metrics"""
        predictions = [
            {f"metric_{i}": 0.5 for i in range(20)},
            {f"metric_{i}": 0.6 for i in range(20)}
        ]

        result = ScoreAggregator.average_scores(predictions)

        assert len(result) == 20
        for i in range(20):
            assert result[f"metric_{i}"] == pytest.approx(0.55)

    def test_average_scores_realistic_scenario(self):
        """Verify averaging in realistic video frame scenario"""
        # Simulating 3 frames from a video analyzed for violence
        frame_predictions = [
            {"violence": 0.2, "firearm": 0.1, "knife": 0.0},  # Frame 1: Safe
            {"violence": 0.8, "firearm": 0.9, "knife": 0.5},  # Frame 2: Weapon detected
            {"violence": 0.3, "firearm": 0.2, "knife": 0.1}  # Frame 3: Safe
        ]

        result = ScoreAggregator.average_scores(frame_predictions)

        assert result["violence"] == pytest.approx((0.2 + 0.8 + 0.3) / 3)
        assert result["firearm"] == pytest.approx((0.1 + 0.9 + 0.2) / 3)
        assert result["knife"] == pytest.approx((0.0 + 0.5 + 0.1) / 3)

    def test_average_scores_maintains_metric_order(self):
        """Verify all metrics from first prediction are present in result"""
        predictions = [
            {"metric_a": 0.1, "metric_b": 0.2, "metric_c": 0.3},
            {"metric_a": 0.4, "metric_b": 0.5, "metric_c": 0.6}
        ]

        result = ScoreAggregator.average_scores(predictions)

        assert set(result.keys()) == {"metric_a", "metric_b", "metric_c"}