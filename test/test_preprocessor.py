from preprocessor import (
    PreprocessedContent,
    PreprocessedText,
    PreprocessedImage,
    PreprocessedVideo,
    ContentPreprocessor,
    TextPreprocessor,
    ImagePreprocessor,
    VideoPreprocessor,
)


class TestPreprocessedDataclasses:
    """Test preprocessed content dataclasses"""

    def test_preprocessed_text_creation(self):
        """Verify PreprocessedText can be created with data"""
        data = [1, 2, 3, 4, 5]
        preprocessed = PreprocessedText(data=data)

        assert preprocessed.data == data
        assert isinstance(preprocessed, PreprocessedContent)

    def test_preprocessed_image_creation(self):
        """Verify PreprocessedImage can be created with data"""
        data = [1] * 16
        original_bytes = b"image_bytes"
        preprocessed = PreprocessedImage(data=data, original_bytes=original_bytes)

        assert preprocessed.data == data
        assert preprocessed.original_bytes == original_bytes
        assert isinstance(preprocessed, PreprocessedContent)

    def test_preprocessed_image_without_original_bytes(self):
        """Verify PreprocessedImage works without original_bytes"""
        data = [1] * 16
        preprocessed = PreprocessedImage(data=data)

        assert preprocessed.data == data
        assert preprocessed.original_bytes is None

    def test_preprocessed_video_creation(self):
        """Verify PreprocessedVideo can be created with frames"""
        frame1 = PreprocessedImage(data=[1] * 16)
        frame2 = PreprocessedImage(data=[2] * 16)
        frames = [frame1, frame2]

        video = PreprocessedVideo(frames=frames)

        assert video.frames == frames
        assert len(video.frames) == 2
        assert isinstance(video, PreprocessedContent)

    def test_preprocessed_video_empty_frames(self):
        """Verify PreprocessedVideo can be created with empty frames"""
        video = PreprocessedVideo(frames=[])

        assert video.frames == []
        assert len(video.frames) == 0


class TestTextPreprocessor:
    """Test TextPreprocessor"""

    def test_text_preprocessor_returns_preprocessed_text(self):
        """Verify TextPreprocessor returns PreprocessedText"""
        preprocessor = TextPreprocessor()
        text = "Hello, this is a test."

        result = preprocessor.preprocess(text)

        assert isinstance(result, PreprocessedText)

    def test_text_preprocessor_generates_numeric_data(self):
        """Verify TextPreprocessor generates numeric data"""
        preprocessor = TextPreprocessor()
        text = "Test text"

        result = preprocessor.preprocess(text)

        assert isinstance(result.data, list)
        assert len(result.data) == 16
        assert all(isinstance(x, int) for x in result.data)

    def test_text_preprocessor_data_in_valid_range(self):
        """Verify TextPreprocessor generates values in expected range"""
        preprocessor = TextPreprocessor()
        text = "Test"

        result = preprocessor.preprocess(text)

        assert all(1 <= x <= 100 for x in result.data)

    def test_text_preprocessor_with_empty_string(self):
        """Verify TextPreprocessor works with empty string"""
        preprocessor = TextPreprocessor()

        result = preprocessor.preprocess("")

        assert isinstance(result, PreprocessedText)
        assert len(result.data) == 16

    def test_text_preprocessor_with_long_text(self):
        """Verify TextPreprocessor works with long text"""
        preprocessor = TextPreprocessor()
        long_text = "a" * 10000

        result = preprocessor.preprocess(long_text)

        assert isinstance(result, PreprocessedText)
        assert len(result.data) == 16


class TestImagePreprocessor:
    """Test ImagePreprocessor"""

    def test_image_preprocessor_returns_preprocessed_image(self):
        """Verify ImagePreprocessor returns PreprocessedImage"""
        preprocessor = ImagePreprocessor()
        image_bytes = b"fake_image_data"

        result = preprocessor.preprocess(image_bytes)

        assert isinstance(result, PreprocessedImage)

    def test_image_preprocessor_generates_numeric_data(self):
        """Verify ImagePreprocessor generates numeric data"""
        preprocessor = ImagePreprocessor()
        image_bytes = b"image"

        result = preprocessor.preprocess(image_bytes)

        assert isinstance(result.data, list)
        assert len(result.data) == 16
        assert all(isinstance(x, int) for x in result.data)

    def test_image_preprocessor_data_in_valid_range(self):
        """Verify ImagePreprocessor generates values in expected range"""
        preprocessor = ImagePreprocessor()
        image_bytes = b"test"

        result = preprocessor.preprocess(image_bytes)

        assert all(1 <= x <= 100 for x in result.data)

    def test_image_preprocessor_with_empty_bytes(self):
        """Verify ImagePreprocessor works with empty bytes"""
        preprocessor = ImagePreprocessor()

        result = preprocessor.preprocess(b"")

        assert isinstance(result, PreprocessedImage)
        assert len(result.data) == 16

    def test_image_preprocessor_with_large_bytes(self):
        """Verify ImagePreprocessor works with large image data"""
        preprocessor = ImagePreprocessor()
        large_image = b"x" * 1000000  # 1MB

        result = preprocessor.preprocess(large_image)

        assert isinstance(result, PreprocessedImage)
        assert len(result.data) == 16


class TestVideoPreprocessor:
    """Test VideoPreprocessor"""

    def test_video_preprocessor_returns_preprocessed_video(self):
        """Verify VideoPreprocessor returns PreprocessedVideo"""
        preprocessor = VideoPreprocessor()
        frames = [b"frame1", b"frame2", b"frame3"]

        result = preprocessor.preprocess(frames)

        assert isinstance(result, PreprocessedVideo)

    def test_video_preprocessor_processes_all_frames(self):
        """Verify VideoPreprocessor processes all frames"""
        preprocessor = VideoPreprocessor()
        frames = [b"frame1", b"frame2", b"frame3"]

        result = preprocessor.preprocess(frames)

        assert len(result.frames) == 3
        assert all(isinstance(frame, PreprocessedImage) for frame in result.frames)

    def test_video_preprocessor_each_frame_has_numeric_data(self):
        """Verify each video frame has numeric data"""
        preprocessor = VideoPreprocessor()
        frames = [b"frame1", b"frame2"]

        result = preprocessor.preprocess(frames)

        for frame in result.frames:
            assert isinstance(frame.data, list)
            assert len(frame.data) == 16
            assert all(isinstance(x, int) for x in frame.data)

    def test_video_preprocessor_single_frame(self):
        """Verify VideoPreprocessor works with single frame"""
        preprocessor = VideoPreprocessor()
        frames = [b"single_frame"]

        result = preprocessor.preprocess(frames)

        assert len(result.frames) == 1
        assert isinstance(result.frames[0], PreprocessedImage)

    def test_video_preprocessor_many_frames(self):
        """Verify VideoPreprocessor works with many frames"""
        preprocessor = VideoPreprocessor()
        frames = [b"frame" for _ in range(100)]

        result = preprocessor.preprocess(frames)

        assert len(result.frames) == 100
        assert all(isinstance(frame, PreprocessedImage) for frame in result.frames)

    def test_video_preprocessor_empty_frames(self):
        """Verify VideoPreprocessor works with empty frames list"""
        preprocessor = VideoPreprocessor()

        result = preprocessor.preprocess([])

        assert isinstance(result, PreprocessedVideo)
        assert len(result.frames) == 0


class TestPreprocessorIntegration:
    """Test preprocessor integration scenarios"""

    def test_text_to_image_to_video_consistency(self):
        """Verify data structure consistency across modalities"""
        text_preprocessor = TextPreprocessor()
        image_preprocessor = ImagePreprocessor()
        video_preprocessor = VideoPreprocessor()

        text_result = text_preprocessor.preprocess("test")
        image_result = image_preprocessor.preprocess(b"test")
        video_result = video_preprocessor.preprocess([b"test", b"test"])

        # All should have 16 numeric values per frame
        assert len(text_result.data) == 16
        assert len(image_result.data) == 16
        assert len(video_result.frames[0].data) == 16

    def test_video_with_image_preprocessor_frames(self):
        """Verify video can be built from image preprocessor frames"""
        image_preprocessor = ImagePreprocessor()

        frame1 = image_preprocessor.preprocess(b"frame1")
        frame2 = image_preprocessor.preprocess(b"frame2")

        video = PreprocessedVideo(frames=[frame1, frame2])

        assert len(video.frames) == 2
        assert all(isinstance(f, PreprocessedImage) for f in video.frames)