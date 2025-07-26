#!/usr/bin/env python3
"""
Test script for the OpenAI-compatible chat API.
"""

import asyncio
import aiohttp
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8080"

async def test_health():
    """Test the health endpoint."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/health") as response:
            result = await response.json()
            print(f"Health check: {result}")
            return response.status == 200

async def test_list_models():
    """Test the models endpoint."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/v1/models") as response:
            result = await response.json()
            print(f"Models: {json.dumps(result, indent=2)}")
            return response.status == 200

async def test_chat_completion():
    """Test non-streaming chat completion."""
    payload = {
        "model": "default-agent",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "stream": False
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            result = await response.json()
            print(f"Chat completion: {json.dumps(result, indent=2)}")
            return response.status == 200

async def test_streaming_chat_completion():
    """Test streaming chat completion."""
    payload = {
        "model": "default-agent",
        "messages": [
            {"role": "user", "content": "Tell me a short story"}
        ],
        "stream": True
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            print("Streaming response:")
            async for line in response.content:
                line_str = line.decode('utf-8').strip()
                if line_str.startswith('data: '):
                    data = line_str[6:]  # Remove 'data: ' prefix
                    if data == '[DONE]':
                        print("Stream completed")
                        break
                    try:
                        chunk = json.loads(data)
                        if chunk.get('choices') and chunk['choices'][0].get('delta', {}).get('content'):
                            content = chunk['choices'][0]['delta']['content']
                            print(content, end='', flush=True)
                    except json.JSONDecodeError:
                        pass
            print()  # New line after streaming
            return response.status == 200

async def test_agent_management():
    """Test agent management endpoints."""
    async with aiohttp.ClientSession() as session:
        # List agents
        async with session.get(f"{BASE_URL}/agents") as response:
            result = await response.json()
            print(f"Agents: {json.dumps(result, indent=2)}")
        
        # Register a new agent
        async with session.post(
            f"{BASE_URL}/agents/register",
            params={"model_id": "test-agent", "agent_url": "http://localhost:8002"}
        ) as response:
            result = await response.json()
            print(f"Register agent: {result}")
        
        # Get agent status
        async with session.get(f"{BASE_URL}/agents/test-agent/status") as response:
            result = await response.json()
            print(f"Agent status: {json.dumps(result, indent=2)}")
        
        # Unregister agent
        async with session.delete(f"{BASE_URL}/agents/test-agent") as response:
            result = await response.json()
            print(f"Unregister agent: {result}")

async def main():
    """Run all tests."""
    print("Testing EgoStarSea Agents API...")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health),
        ("List Models", test_list_models),
        ("Chat Completion", test_chat_completion),
        ("Streaming Chat", test_streaming_chat_completion),
        ("Agent Management", test_agent_management),
    ]
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        try:
            success = await test_func()
            print(f"✅ {test_name} {'passed' if success else 'failed'}")
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
    
    print("\n" + "=" * 50)
    print("Testing completed!")

if __name__ == "__main__":
    asyncio.run(main())
