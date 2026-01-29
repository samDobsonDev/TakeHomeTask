import asyncio
import json
import base64
import requests
from pathlib import Path
from src.request_handler import RequestHandler
from src.content_loader import ContentLoader


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
        print(json.dumps(moderation_response, indent=2))
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
        print(json.dumps(moderation_response, indent=2))
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
        print(json.dumps(moderation_response, indent=2))
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
        # Create 3 frames (same image repeated for demo)
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
        print(json.dumps(moderation_response, indent=2))
    except FileNotFoundError as e:
        print(f"Local video file not found: {e}")
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
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())