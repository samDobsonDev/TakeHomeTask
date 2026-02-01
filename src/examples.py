import asyncio
import json
import base64
import requests
import os
from pathlib import Path
from typing import Union
from dotenv import load_dotenv
from src.request_handler import RequestHandler, ServiceContainer, ModerationResponse, ErrorResponse, VideoModelResult
from src.content_loader import ContentLoader
from src.model import RandomViolenceModel, RandomHateSpeechModel
from src.open_ai.openai_models import OpenAIViolenceModel, OpenAIHateSpeechModel


# Load environment variables from .env file
load_dotenv()


def print_moderation_response(response: Union[ModerationResponse, ErrorResponse]) -> None:
    """Pretty print a moderation response with all available fields"""
    print("\n" + "-" * 60)
    if isinstance(response, ErrorResponse):
        print("Status: error")
        print(f"Error: {response.error}")
        print(f"Status Code: {response.status_code}")
    else:
        print("Status: success")
        print(f"\nResults ({len(response.results)} categories):")
        for category, result in response.results.items():
            print(f"\n  {category.upper()}:")
            print(f"    Risk Level: {result.risk_level}")
            if len(result.models) == 1:
                # Single model
                model = result.models[0]
                print(f"    Model: {model.model_name}")
                if isinstance(model, VideoModelResult):
                    print(f"    Video Frames: {len(model.frames)}")
                    for frame in model.frames:
                        print(f"      Frame {frame.frame}: risk={frame.risk_level}, scores={frame.scores}")
                else:
                    print(f"    Scores: {model.scores}")
            else:
                # Multiple models
                print(f"    Multiple Models: {len(result.models)}")
                for model in result.models:
                    print(f"      - {model.model_name}: risk={model.risk_level}")
                    if isinstance(model, VideoModelResult):
                        print(f"        Frames: {len(model.frames)}")
                        for frame in model.frames:
                            print(f"          Frame {frame.frame}: risk={frame.risk_level}, scores={frame.scores}")
                    else:
                        print(f"        Scores: {model.scores}")
    print("-" * 60 + "\n")


async def moderate_text():
    """Example 1: Moderate text content"""
    print("=" * 60)
    print("Example 1: Moderate Text Content")
    print("=" * 60)
    try:
        handler = RequestHandler()
        text_content = "This is a sample text for moderation testing."
        request_json = json.dumps({
            "content": text_content,
            "modality": "text",
            "customer": "test_customer"
        })
        moderation_response = await handler.handle_moderate_request(request_json)
        print_moderation_response(moderation_response)
    except Exception as e:
        print(f"Error: {e}")


async def moderate_image_from_url():
    """Example 2: Download image from URL and moderate"""
    print("\n" + "=" * 60)
    print("Example 2: Moderate Image from URL")
    print("=" * 60)
    image_url = "https://t3.ftcdn.net/jpg/03/21/62/56/360_F_321625657_rauGwvaYjtbETuwxn9kpBWKDYrVUMdB4.jpg"
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        base64_image = base64.b64encode(response.content).decode('utf-8')
        handler = RequestHandler()
        request_json = json.dumps({
            "content": base64_image,
            "modality": "image",
            "customer": "test_customer"
        })
        moderation_response = await handler.handle_moderate_request(request_json)
        print_moderation_response(moderation_response)
    except requests.RequestException as e:
        print(f"Failed to download image: {e}")
    except Exception as e:
        print(f"Error: {e}")


async def moderate_image_from_local_file():
    """Example 3: Load image from local resources and moderate"""
    print("\n" + "=" * 60)
    print("Example 3: Moderate Image from Local File")
    print("=" * 60)
    try:
        image_path = Path(__file__).parent.parent / "resources" / "gun.png"
        image_bytes = ContentLoader.load_image(image_path)
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        handler = RequestHandler()
        request_json = json.dumps({
            "content": base64_image,
            "modality": "image",
            "customer": "test_customer"
        })
        moderation_response = await handler.handle_moderate_request(request_json)
        print_moderation_response(moderation_response)
    except FileNotFoundError as e:
        print(f"Local image file not found: {e}")
    except Exception as e:
        print(f"Error: {e}")


async def moderate_video():
    """Example 4: Moderate video content with multiple frames"""
    print("\n" + "=" * 60)
    print("Example 4: Moderate Video Content")
    print("=" * 60)
    try:
        image_path = Path(__file__).parent.parent / "resources" / "gun.png"
        image_bytes = ContentLoader.load_image(image_path)
        frame1_b64 = base64.b64encode(image_bytes).decode('utf-8')
        frame2_b64 = base64.b64encode(image_bytes).decode('utf-8')
        frame3_b64 = base64.b64encode(image_bytes).decode('utf-8')
        handler = RequestHandler()
        request_json = json.dumps({
            "content": [frame1_b64, frame2_b64, frame3_b64],
            "modality": "video",
            "customer": "test_customer"
        })
        moderation_response = await handler.handle_moderate_request(request_json)
        print_moderation_response(moderation_response)
    except FileNotFoundError as e:
        print(f"Local video file not found: {e}")
    except Exception as e:
        print(f"Error: {e}")


async def moderate_violence_with_dual_hate_speech_models():
    """Example 5: Moderate hate speech content with custom container using two hate speech models"""
    print("\n" + "=" * 60)
    print("Example 5: Text with Dual Hate Speech Models (Random + OpenAI)")
    print("=" * 60)
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY environment variable not set. Please set it in your .env file.")
            return
        container = ServiceContainer(
            models=[
                RandomHateSpeechModel(),
                OpenAIHateSpeechModel(api_key=api_key),
            ]
        )
        handler = RequestHandler(container=container)
        text_content = "Fuck you I hate you all!"
        request_json = json.dumps({
            "content": text_content,
            "modality": "text",
            "customer": "test_customer"
        })
        moderation_response = await handler.handle_moderate_request(request_json)
        print_moderation_response(moderation_response)
    except Exception as e:
        print(f"Error: {e}")


async def moderate_video_with_dual_violence_models():
    """Example 6: Moderate video with dual violence models (Random + OpenAI)"""
    print("\n" + "=" * 60)
    print("Example 6: Video with Dual Violence Models (Random + OpenAI)")
    print("=" * 60)
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY environment variable not set. Please set it in your .env file.")
            return
        image_path = Path(__file__).parent.parent / "resources" / "gun.png"
        image_bytes = ContentLoader.load_image(image_path)
        frame1_b64 = base64.b64encode(image_bytes).decode('utf-8')
        frame2_b64 = base64.b64encode(image_bytes).decode('utf-8')
        frame3_b64 = base64.b64encode(image_bytes).decode('utf-8')
        container = ServiceContainer(
            models=[
                RandomViolenceModel(),
                OpenAIViolenceModel(api_key=api_key),
            ]
        )
        handler = RequestHandler(container=container)
        request_json = json.dumps({
            "content": [frame1_b64, frame2_b64, frame3_b64],
            "modality": "video",
            "customer": "test_customer"
        })
        moderation_response = await handler.handle_moderate_request(request_json)
        print_moderation_response(moderation_response)
    except FileNotFoundError as e:
        print(f"Local image file not found: {e}")
    except Exception as e:
        print(f"Error: {e}")


async def main():
    """Run all moderation examples"""
    print("\n" + "=" * 60)
    print("Content Moderation API Examples")
    print("=" * 60)
    # Example 1: Text moderation
    await moderate_text()
    # Example 2: Image from URL
    await moderate_image_from_url()
    # Example 3: Image from local file
    await moderate_image_from_local_file()
    # Example 4: Video moderation
    await moderate_video()
    # Example 5: Text hate speech detection with dual models
    await moderate_violence_with_dual_hate_speech_models()
    # Example 6: Video violence detection with dual models
    await moderate_video_with_dual_violence_models()
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())