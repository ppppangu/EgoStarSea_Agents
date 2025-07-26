#!/usr/bin/env python3
"""
Test script for the multimodal chat API.
"""

import asyncio
import aiohttp
import json
import base64
from typing import Dict, Any

BASE_URL = "http://localhost:8081"

async def test_multimodal_text_only():
    """测试纯文本聊天"""
    payload = {
        "model": "gpt-4o-mini",
        "user": "test-user-123",
        "messages": [
            {"role": "user", "content": "你好，请简单介绍一下自己"}
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
            print(f"Text-only chat: {json.dumps(result, indent=2, ensure_ascii=False)}")
            return response.status == 200

async def test_multimodal_with_audio_response():
    """测试返回语音的聊天"""
    payload = {
        "model": "gpt-4o-mini",
        "user": "test-user-456",
        "messages": [
            {"role": "user", "content": "请说一句简短的问候语"}
        ],
        "return_audio": True
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{BASE_URL}/v1/chat/multimodal",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            result = await response.json()
            
            # 检查是否包含音频响应
            has_audio = False
            if result.get('choices') and result['choices'][0].get('audio_base64'):
                has_audio = True
                audio_length = len(result['choices'][0]['audio_base64'])
                print(f"Audio response received, length: {audio_length} characters")
                # 隐藏音频内容以减少输出
                result['choices'][0]['audio_base64'] = f"[AUDIO_DATA_{audio_length}_CHARS]"
            
            print(f"Chat with audio response: {json.dumps(result, indent=2, ensure_ascii=False)}")
            print(f"Has audio response: {has_audio}")
            return response.status == 200

async def test_multimodal_with_fake_audio_input():
    """测试带音频输入的聊天（使用假音频数据）"""
    # 创建假的音频 base64 数据（实际使用时应该是真实的音频文件）
    fake_audio = base64.b64encode(b"fake audio data").decode('utf-8')
    
    payload = {
        "model": "gpt-4o-mini",
        "user": "test-user-789",
        "audio_base64": fake_audio,
        "messages": [
            {"role": "user", "content": "这是文本输入"}
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
            print(f"Chat with audio input: {json.dumps(result, indent=2, ensure_ascii=False)}")
            return response.status == 200

async def test_memory_persistence():
    """测试记忆持久化"""
    user_id = "memory-test-user"
    
    # 第一次对话
    payload1 = {
        "model": "gpt-4o-mini",
        "user": user_id,
        "messages": [
            {"role": "user", "content": "我叫张三，喜欢吃苹果"}
        ]
    }
    
    # 第二次对话
    payload2 = {
        "model": "gpt-4o-mini",
        "user": user_id,
        "messages": [
            {"role": "user", "content": "你还记得我的名字和喜好吗？"}
        ]
    }
    
    async with aiohttp.ClientSession() as session:
        # 第一次对话
        async with session.post(
            f"{BASE_URL}/v1/chat/multimodal",
            json=payload1,
            headers={"Content-Type": "application/json"}
        ) as response1:
            result1 = await response1.json()
            print("First conversation:")
            print(f"  {json.dumps(result1, indent=2, ensure_ascii=False)}")
        
        # 第二次对话
        async with session.post(
            f"{BASE_URL}/v1/chat/multimodal",
            json=payload2,
            headers={"Content-Type": "application/json"}
        ) as response2:
            result2 = await response2.json()
            print("Second conversation (testing memory):")
            print(f"  {json.dumps(result2, indent=2, ensure_ascii=False)}")
            
            return response1.status == 200 and response2.status == 200

async def main():
    """Run all multimodal tests."""
    print("Testing EgoStarSea Agents Multimodal API...")
    print("=" * 50)
    
    tests = [
        ("Text-only Chat", test_multimodal_text_only),
        ("Chat with Audio Response", test_multimodal_with_audio_response),
        ("Chat with Audio Input", test_multimodal_with_fake_audio_input),
        ("Memory Persistence", test_memory_persistence),
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
    print("Multimodal testing completed!")

if __name__ == "__main__":
    asyncio.run(main())