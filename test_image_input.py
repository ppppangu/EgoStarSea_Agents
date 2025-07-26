#!/usr/bin/env python3
"""
Test script for image input in the multimodal chat API.
"""

import asyncio
import aiohttp
import json
import base64

BASE_URL = "http://localhost:8081"

# Create a simple test image (1x1 pixel red PNG)
def create_test_image_base64():
    """Create a minimal test image in base64 format"""
    # This is a 1x1 red pixel PNG image in base64
    return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

async def test_image_input():
    """Test image input with multimodal chat"""
    image_base64 = create_test_image_base64()
    
    payload = {
        "model": "gpt-4o-mini",
        "user": "test-user-image",
        "image_base64": image_base64,
        "messages": [
            {"role": "user", "content": "What do you see in this image?"}
        ],
        "return_audio": False
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{BASE_URL}/v1/chat/multimodal",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            result = await response.json()
            print(f"Image input test: {json.dumps(result, indent=2, ensure_ascii=False)}")
            return response.status == 200

async def test_image_url():
    """Test image URL input"""
    payload = {
        "model": "gpt-4o-mini", 
        "user": "test-user-url",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
        "messages": [
            {"role": "user", "content": "Describe this landscape image"}
        ],
        "return_audio": False
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{BASE_URL}/v1/chat/multimodal",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            result = await response.json()
            print(f"Image URL test: {json.dumps(result, indent=2, ensure_ascii=False)}")
            return response.status == 200

async def main():
    print("Testing Image Input for Multimodal API...")
    print("=" * 50)
    
    print("\nImage Base64 Input:")
    print("-" * 30)
    success1 = await test_image_input()
    print(f"✅ Image Base64 Input {'passed' if success1 else 'failed'}")
    
    print("\nImage URL Input:")
    print("-" * 30)
    success2 = await test_image_url()
    print(f"✅ Image URL Input {'passed' if success2 else 'failed'}")
    
    print("\n" + "=" * 50)
    print("Image testing completed!")

if __name__ == "__main__":
    asyncio.run(main())
