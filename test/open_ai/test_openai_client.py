import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import BaseModel
from src.open_ai.openai_client import OpenAIClient


# Test Pydantic models for response format
class SampleResponse(BaseModel):
    """Sample response format for testing"""
    score: float
    label: str


class TestOpenAIClientInitialization:
    """Test OpenAIClient initialization"""

    def test_init_with_api_key(self):
        """Test client initializes with provided API key"""
        with patch('src.open_ai.openai_client.OpenAI') as mock_openai:
            client = OpenAIClient(api_key="test_key")
            mock_openai.assert_called_once_with(api_key="test_key")
            assert client.client is not None

    def test_init_without_api_key(self):
        """Test client initializes without API key (uses env var)"""
        with patch('src.open_ai.openai_client.OpenAI') as mock_openai:
            client = OpenAIClient()
            mock_openai.assert_called_once_with(api_key=None)
            assert client.client is not None


class TestOpenAIClientAnalyzeText:
    """Test analyze_text method"""

    @pytest.mark.asyncio
    async def test_analyze_text_success(self):
        """Test successful text analysis"""
        with patch('src.open_ai.openai_client.OpenAI') as mock_openai_class:
            # Setup mock
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.output_parsed = SampleResponse(score=0.8, label="high")
            mock_client.responses.parse.return_value = mock_response

            # Create client and call method
            client = OpenAIClient(api_key="test_key")
            result = await client.analyze_text(
                text="test content",
                model="gpt-4o-2024-08-06",
                prompt="Analyze this:",
                response_format=SampleResponse
            )

            # Assertions
            assert result.score == 0.8
            assert result.label == "high"
            mock_client.responses.parse.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_text_prompt_formatting(self):
        """Test that prompt and text are properly formatted"""
        with patch('src.open_ai.openai_client.OpenAI') as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.output_parsed = SampleResponse(score=0.5, label="medium")
            mock_client.responses.parse.return_value = mock_response

            client = OpenAIClient(api_key="test_key")
            await client.analyze_text(
                text="malicious content",
                model="gpt-4o",
                prompt="Check for toxicity:",
                response_format=SampleResponse
            )

            # Verify the call structure
            call_args = mock_client.responses.parse.call_args
            content = call_args[1]['input'][0]['content'][0]['text']

            # Assert both prompt and text are in the formatted message
            assert "Check for toxicity:" in content
            assert "malicious content" in content


class TestOpenAIClientAnalyzeImage:
    """Test analyze_image method"""

    @pytest.mark.asyncio
    async def test_analyze_image_success(self):
        """Test successful image analysis"""
        with patch('src.open_ai.openai_client.OpenAI') as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.output_parsed = SampleResponse(score=0.9, label="explicit")
            mock_client.responses.parse.return_value = mock_response

            client = OpenAIClient(api_key="test_key")
            image_bytes = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100  # Minimal PNG-like bytes

            result = await client.analyze_image(
                image_bytes=image_bytes,
                model="gpt-4o-2024-08-06",
                prompt="Analyze image:",
                response_format=SampleResponse
            )

            assert result.score == 0.9
            assert result.label == "explicit"
            mock_client.responses.parse.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_image_base64_encoding(self):
        """Test that image bytes are properly base64 encoded"""
        with patch('src.open_ai.openai_client.OpenAI') as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.output_parsed = SampleResponse(score=0.5, label="safe")
            mock_client.responses.parse.return_value = mock_response

            client = OpenAIClient(api_key="test_key")
            image_bytes = b'test_image_data'

            await client.analyze_image(
                image_bytes=image_bytes,
                model="gpt-4o",
                prompt="Check image:",
                response_format=SampleResponse
            )

            # Verify base64 encoding in the call
            call_args = mock_client.responses.parse.call_args
            image_url = call_args[1]['input'][0]['content'][1]['image_url']

            # Should contain base64 encoded data
            assert "data:image/jpeg;base64," in image_url
            assert "dGVzdF9pbWFnZV9kYXRh" in image_url  # base64 of 'test_image_data'

    @pytest.mark.asyncio
    async def test_analyze_image_request_structure(self):
        """Test the request structure for image analysis"""
        with patch('src.open_ai.openai_client.OpenAI') as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.output_parsed = SampleResponse(score=0.3, label="low")
            mock_client.responses.parse.return_value = mock_response

            client = OpenAIClient(api_key="test_key")
            await client.analyze_image(
                image_bytes=b'image',
                model="gpt-4o",
                prompt="Analyze:",
                response_format=SampleResponse
            )

            call_kwargs = mock_client.responses.parse.call_args[1]

            # Verify structure
            assert call_kwargs['model'] == "gpt-4o"
            assert call_kwargs['text_format'] == SampleResponse
            assert len(call_kwargs['input'][0]['content']) == 2  # text + image


class TestOpenAIClientAnalyzeVideo:
    """Test analyze_video method"""

    @pytest.mark.asyncio
    async def test_analyze_video_success(self):
        """Test successful video (frames) analysis"""
        with patch('src.open_ai.openai_client.OpenAI') as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client

            # Setup multiple responses for multiple frames
            mock_response_1 = MagicMock()
            mock_response_1.output_parsed = SampleResponse(score=0.7, label="medium")

            mock_response_2 = MagicMock()
            mock_response_2.output_parsed = SampleResponse(score=0.8, label="high")

            mock_client.responses.parse.side_effect = [mock_response_1, mock_response_2]

            client = OpenAIClient(api_key="test_key")
            frames = [b'frame1', b'frame2']

            results = await client.analyze_video(
                frames=frames,
                model="gpt-4o",
                prompt="Analyze:",
                response_format=SampleResponse
            )

            assert len(results) == 2
            assert results[0].score == 0.7
            assert results[1].score == 0.8

    @pytest.mark.asyncio
    async def test_analyze_video_processes_all_frames(self):
        """Test that all frames are processed"""
        with patch('src.open_ai.openai_client.OpenAI') as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client

            # Create mock responses for 5 frames
            mock_responses = [
                MagicMock(output_parsed=SampleResponse(score=float(i) * 0.1, label=f"frame{i}"))
                for i in range(5)
            ]
            mock_client.responses.parse.side_effect = mock_responses

            client = OpenAIClient(api_key="test_key")
            frames = [b'frame' + str(i).encode() for i in range(5)]

            results = await client.analyze_video(
                frames=frames,
                model="gpt-4o",
                prompt="Analyze:",
                response_format=SampleResponse
            )

            # Verify all frames were processed
            assert len(results) == 5
            assert mock_client.responses.parse.call_count == 5

    @pytest.mark.asyncio
    async def test_analyze_video_empty_frames(self):
        """Test video analysis with empty frames list"""
        with patch('src.open_ai.openai_client.OpenAI') as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client

            client = OpenAIClient(api_key="test_key")

            results = await client.analyze_video(
                frames=[],
                model="gpt-4o",
                prompt="Analyze:",
                response_format=SampleResponse
            )

            assert results == []
            mock_client.responses.parse.assert_not_called()