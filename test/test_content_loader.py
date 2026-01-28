import pytest
import tempfile
from pathlib import Path
from src.content_loader import ContentLoader, ContentLoadError


@pytest.fixture
def temp_image_file():
    """Create a temporary image file for testing"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
        f.write(b"fake_image_data")
        temp_path = f.name
    yield temp_path
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_video_file():
    """Create a temporary video file for testing"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        f.write(b"fake_video_data")
        temp_path = f.name
    yield temp_path
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_empty_file():
    """Create an empty temporary file for testing"""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_path = f.name
    yield temp_path
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


class TestContentLoaderLoadImage:
    """Test ContentLoader.load_image() method"""

    def test_load_image_from_valid_file(self, temp_image_file):
        """Verify loading image from valid file path"""
        result = ContentLoader.load_image(temp_image_file)

        assert result == b"fake_image_data"
        assert isinstance(result, bytes)

    def test_load_image_returns_bytes(self, temp_image_file):
        """Verify load_image returns bytes object"""
        result = ContentLoader.load_image(temp_image_file)

        assert isinstance(result, bytes)

    def test_load_image_nonexistent_file(self):
        """Verify loading from nonexistent file raises ContentLoadError"""
        with pytest.raises(ContentLoadError, match="Image file not found"):
            ContentLoader.load_image("/nonexistent/path/image.jpg")

    def test_load_image_directory_path(self):
        """Verify loading from directory path raises ContentLoadError"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ContentLoadError, match="Path is not a file"):
                ContentLoader.load_image(tmpdir)

    def test_load_image_empty_file(self, temp_empty_file):
        """Verify loading empty file raises ContentLoadError"""
        with pytest.raises(ContentLoadError, match="Image file is empty"):
            ContentLoader.load_image(temp_empty_file)

    def test_load_image_large_file(self):
        """Verify loading large image file"""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            large_data = b"x" * 1000000  # 1MB
            f.write(large_data)
            temp_path = f.name

        try:
            result = ContentLoader.load_image(temp_path)
            assert len(result) == 1000000
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_image_with_string_path(self, temp_image_file):
        """Verify load_image works with string path"""
        result = ContentLoader.load_image(temp_image_file)  # String path
        assert isinstance(result, bytes)

    def test_load_image_with_path_object(self, temp_image_file):
        """Verify load_image works with Path object"""
        path_obj = Path(temp_image_file)
        result = ContentLoader.load_image(path_obj)
        assert isinstance(result, bytes)

    def test_load_image_preserves_binary_data(self):
        """Verify binary data is preserved exactly"""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            original_data = b"\x00\x01\x02\xff\xfe\xfd"
            f.write(original_data)
            temp_path = f.name

        try:
            result = ContentLoader.load_image(temp_path)
            assert result == original_data
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestContentLoaderLoadVideo:
    """Test ContentLoader.load_video() method"""

    def test_load_video_from_valid_file(self, temp_video_file):
        """Verify loading video from valid file path"""
        result = ContentLoader.load_video(temp_video_file)

        assert isinstance(result, list)
        assert len(result) == 10
        assert all(frame == b"fake_video_data" for frame in result)

    def test_load_video_returns_list_of_bytes(self, temp_video_file):
        """Verify load_video returns list of bytes objects"""
        result = ContentLoader.load_video(temp_video_file)

        assert isinstance(result, list)
        assert all(isinstance(frame, bytes) for frame in result)

    def test_load_video_replicates_frames(self, temp_video_file):
        """Verify video replicates single frame 10 times"""
        result = ContentLoader.load_video(temp_video_file)

        assert len(result) == 10
        assert result[0] == result[1] == result[9]

    def test_load_video_nonexistent_file(self):
        """Verify loading video from nonexistent file raises ContentLoadError"""
        with pytest.raises(ContentLoadError, match="Image file not found"):
            ContentLoader.load_video("/nonexistent/video.mp4")

    def test_load_video_empty_file(self, temp_empty_file):
        """Verify loading video from empty file raises ContentLoadError"""
        with pytest.raises(ContentLoadError, match="Image file is empty"):
            ContentLoader.load_video(temp_empty_file)

    def test_load_video_with_large_file(self):
        """Verify loading large video file"""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            large_data = b"x" * 1000000
            f.write(large_data)
            temp_path = f.name

        try:
            result = ContentLoader.load_video(temp_path)
            assert len(result) == 10
            assert all(len(frame) == 1000000 for frame in result)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_video_with_string_path(self, temp_video_file):
        """Verify load_video works with string path"""
        result = ContentLoader.load_video(temp_video_file)
        assert isinstance(result, list)
        assert len(result) == 10

    def test_load_video_with_path_object(self, temp_video_file):
        """Verify load_video works with Path object"""
        path_obj = Path(temp_video_file)
        result = ContentLoader.load_video(path_obj)
        assert isinstance(result, list)
        assert len(result) == 10


class TestContentLoadError:
    """Test ContentLoadError exception"""

    def test_content_load_error_is_exception(self):
        """Verify ContentLoadError is an Exception"""
        error = ContentLoadError("test error")
        assert isinstance(error, Exception)

    def test_content_load_error_message(self):
        """Verify ContentLoadError preserves error message"""
        message = "This is a test error"
        error = ContentLoadError(message)
        assert str(error) == message


class TestContentLoaderErrorMessages:
    """Test that error messages are clear and helpful"""

    def test_error_includes_file_path(self):
        """Verify error messages include the file path"""
        bad_path = "/path/to/nonexistent/file.jpg"
        with pytest.raises(ContentLoadError) as exc_info:
            ContentLoader.load_image(bad_path)
        error_message = str(exc_info.value).lower()
        assert "file.jpg" in error_message
        assert "not found" in error_message

    def test_error_for_directory_specifies_not_file(self):
        """Verify error message for directory specifies it's not a file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ContentLoadError) as exc_info:
                ContentLoader.load_image(tmpdir)

            assert "not a file" in str(exc_info.value).lower()

    def test_error_for_empty_file_specifies_empty(self, temp_empty_file):
        """Verify error message for empty file specifies it's empty"""
        with pytest.raises(ContentLoadError) as exc_info:
            ContentLoader.load_image(temp_empty_file)

        assert "empty" in str(exc_info.value).lower()