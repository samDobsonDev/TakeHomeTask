import unittest
from unittest.mock import MagicMock
from src.score_calculator import ScoreCalculator


class TestScoreCalculator(unittest.TestCase):
    """Test cases for ScoreCalculator.compute_average_score()"""

    def test_compute_average_score_single_prediction(self):
        """Test computing average from a single text/image prediction"""
        mock_prediction = MagicMock()
        mock_prediction.to_dict.return_value = {"metric1": 0.5, "metric2": 0.3, "metric3": 0.7}
        result = ScoreCalculator.compute_average_score(mock_prediction)
        
        # Average of 0.5, 0.3, 0.7 is 0.5
        self.assertAlmostEqual(result, 0.5, places=5)

    def test_compute_average_score_empty_prediction(self):
        """Test computing average from a prediction with no scores"""
        mock_prediction = MagicMock()
        mock_prediction.to_dict.return_value = {}
        result = ScoreCalculator.compute_average_score(mock_prediction)
        
        self.assertEqual(result, 0.0)

    def test_compute_average_score_video_frames(self):
        """Test computing average from video frames (list of predictions)"""
        mock_frame1 = MagicMock()
        mock_frame1.to_dict.return_value = {"metric1": 0.2, "metric2": 0.4}
        mock_frame2 = MagicMock()
        mock_frame2.to_dict.return_value = {"metric1": 0.6, "metric2": 0.8}
        frames = [mock_frame1, mock_frame2]
        result = ScoreCalculator.compute_average_score(frames)
        
        # Average of all scores: (0.2 + 0.4 + 0.6 + 0.8) / 4 = 0.5
        self.assertAlmostEqual(result, 0.5, places=5)

    def test_compute_average_score_video_single_frame(self):
        """Test computing average from a single video frame"""
        mock_frame = MagicMock()
        mock_frame.to_dict.return_value = {"metric1": 0.3, "metric2": 0.7}
        frames = [mock_frame]
        result = ScoreCalculator.compute_average_score(frames)
        
        # Average of 0.3 and 0.7 is 0.5
        self.assertAlmostEqual(result, 0.5, places=5)

    def test_compute_average_score_video_empty_frames(self):
        """Test computing average from an empty list of frames"""
        frames = []
        result = ScoreCalculator.compute_average_score(frames)
        
        self.assertEqual(result, 0.0)

    def test_compute_average_score_video_frames_with_different_metrics(self):
        """Test computing average from frames with different metrics per frame"""
        mock_frame1 = MagicMock()
        mock_frame1.to_dict.return_value = {"metric1": 0.2, "metric2": 0.8}
        mock_frame2 = MagicMock()
        mock_frame2.to_dict.return_value = {"metric1": 0.4, "metric2": 0.6, "metric3": 0.5}
        frames = [mock_frame1, mock_frame2]
        result = ScoreCalculator.compute_average_score(frames)
        
        # Average of all scores: (0.2 + 0.8 + 0.4 + 0.6 + 0.5) / 5 = 0.5
        self.assertAlmostEqual(result, 0.5, places=5)

    def test_compute_average_score_single_metric(self):
        """Test computing average from a prediction with single metric"""
        mock_prediction = MagicMock()
        mock_prediction.to_dict.return_value = {"metric1": 0.75}
        result = ScoreCalculator.compute_average_score(mock_prediction)
        
        self.assertAlmostEqual(result, 0.75, places=5)

    def test_compute_average_score_zero_values(self):
        """Test computing average with zero values"""
        mock_prediction = MagicMock()
        mock_prediction.to_dict.return_value = {"metric1": 0.0, "metric2": 0.0, "metric3": 0.0}
        result = ScoreCalculator.compute_average_score(mock_prediction)
        
        self.assertEqual(result, 0.0)

    def test_compute_average_score_high_values(self):
        """Test computing average with high values"""
        mock_prediction = MagicMock()
        mock_prediction.to_dict.return_value = {"metric1": 0.9, "metric2": 0.95, "metric3": 1.0}
        result = ScoreCalculator.compute_average_score(mock_prediction)
        
        # Average of 0.9, 0.95, 1.0 is approximately 0.95
        self.assertAlmostEqual(result, 0.95, places=5)
