from pathlib import Path
from typing import Union


class ContentLoadError(Exception):
    """Raised when content fails to load"""
    pass


class ContentLoader:
    """Loads raw content (images, videos) from file paths"""

    @staticmethod
    def load_image(path: Union[str, Path]) -> bytes:
        """
        Load an image from file path.

        Args:
            path: Path to image file

        Returns:
            List containing image bytes

        Raises:
            ContentLoadError: If file cannot be read
        """
        try:
            path = Path(path)
            if not path.exists():
                raise ContentLoadError(f"Image file not found: {path}")
            if not path.is_file():
                raise ContentLoadError(f"Path is not a file: {path}")
            with open(path, "rb") as f:
                image_bytes = f.read()
            if not image_bytes:
                raise ContentLoadError(f"Image file is empty: {path}")
            return image_bytes
        except ContentLoadError:
            raise
        except Exception as e:
            raise ContentLoadError(f"Failed to load image from {path}: {str(e)}")

    @staticmethod
    def load_video(path: Union[str, Path]) -> list[bytes]:
        """
        Load a video from file path.

        For simplicity, treats video as multiple copies of a single frame.

        Args:
            path: Path to video file

        Returns:
            List of video frame bytes (repeated 10 times)

        Raises:
            ContentLoadError: If file cannot be read
        """
        try:
            img_bytes = ContentLoader.load_image(path)
            return [img_bytes] * 10
        except ContentLoadError:
            raise
        except Exception as e:
            raise ContentLoadError(f"Failed to load video from {path}: {str(e)}")