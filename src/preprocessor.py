from abc import ABC, abstractmethod
import random
from dataclasses import dataclass


@dataclass
class PreprocessedContent:
    """Base class for preprocessed content"""
    pass


@dataclass
class PreprocessedText(PreprocessedContent):
    """Preprocessed text content"""
    data: list[int]
    original_text: str


@dataclass
class PreprocessedImage(PreprocessedContent):
    """Preprocessed image content"""
    data: list[int]
    original_bytes: bytes


@dataclass
class PreprocessedVideo(PreprocessedContent):
    """Preprocessed video content as a sequence of frames"""
    frames: list[PreprocessedImage]


class ContentPreprocessor(ABC):
    """
    Abstract base class for preprocessing raw content into numeric format.

    Handles conversion of different content modalities (text, images, videos)
    into standardized format for consumption by ML models.
    """

    @abstractmethod
    def preprocess(self, content: str | bytes | list[bytes]) -> PreprocessedContent:
        """Convert raw content to preprocessed format"""
        pass


class TextPreprocessor(ContentPreprocessor):
    """
    Preprocessor for text content.

    Converts raw text strings into numeric vector representation.
    """

    def preprocess(self, content: str) -> PreprocessedText:
        """Convert text to numeric vector"""
        data = [random.randint(1, 100)] * 16
        return PreprocessedText(data=data, original_text=content)


class FrameBasedPreprocessor(ContentPreprocessor, ABC):
    """
    Base class for frame-based media preprocessing (images and videos).

    Handles the common logic of processing individual frames.
    """

    @staticmethod
    def preprocess_frame(_frame: bytes) -> list[int]:
        """Convert frame bytes to numeric vector"""
        return [random.randint(1, 100)] * 16


class ImagePreprocessor(FrameBasedPreprocessor):
    """
    Preprocessor for image content.

    Converts image bytes into numeric format for model input.
    """

    def preprocess(self, content: bytes) -> PreprocessedImage:
        """Process a single image"""
        data = self.preprocess_frame(content)
        return PreprocessedImage(data=data, original_bytes=content)


class VideoPreprocessor(FrameBasedPreprocessor):
    """
    Preprocessor for video content.

    Treats videos as a sequence of frames, processes each frame separately,
    returning a PreprocessedVideo with multiple PreprocessedImage frames.
    """

    def preprocess(self, content: list[bytes]) -> PreprocessedVideo:
        """Process all frames, returning one PreprocessedImage per frame"""
        frames = [PreprocessedImage(data=self.preprocess_frame(frame), original_bytes=frame) for frame in content]
        return PreprocessedVideo(frames=frames)