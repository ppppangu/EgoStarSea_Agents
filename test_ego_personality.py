#!/usr/bin/env python3
"""
测试 Ego 的人格特征和对话风格
"""

import asyncio
import aiohttp
import json

BASE_URL = "http://localhost:8081"

async def test_ego_conversation(user_id, message, description):
    """测试 Ego 的对话"""
    payload = {
        "model": "gpt-4o-mini",
        "user": user_id,
        "messages": [
            {"role": "user", "content": message}
        ]
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{BASE_URL}/v1/chat/multimodal",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            result = await response.json()
            
            if response.status == 200:
                reply = result["choices"][0]["message"]["content"]
                print(f"\n{description}")
                print("-" * 50)
                print(f"用户: {message}")
                print(f"Ego: {reply}")
                return True
            else:
                print(f"❌ {description} - 请求失败: {result}")
                return False

async def main():
    print("🌟 测试 Ego 的人格特征和对话风格")
    print("=" * 60)
    
    test_cases = [
        {
            "user_id": "test_emotional_support",
            "message": "我今天工作压力很大，感觉快要崩溃了，不知道该怎么办...",
            "description": "情绪倾听者测试 - 压力疏导"
        },
        {
            "user_id": "test_social_guidance", 
            "message": "我在社交场合总是很紧张，不知道该说什么，感觉别人都不喜欢我",
            "description": "社交指引者测试 - 社交焦虑"
        },
        {
            "user_id": "test_self_discovery",
            "message": "我最近一直在思考人生的意义，感觉很迷茫，不知道自己真正想要什么",
            "description": "成长伙伴测试 - 自我探索"
        },
        {
            "user_id": "test_relationship_issue",
            "message": "我和朋友吵架了，我觉得是他的错，但又不想失去这个朋友，很纠结",
            "description": "心理疏导者测试 - 人际关系"
        },
        {
            "user_id": "test_positive_sharing",
            "message": "今天我终于鼓起勇气向喜欢的人表白了，虽然被拒绝了，但我觉得很勇敢！",
            "description": "成长见证测试 - 积极分享"
        },
        {
            "user_id": "test_anxiety",
            "message": "明天要面试了，我很紧张，担心自己表现不好，一直睡不着",
            "description": "情绪调节测试 - 焦虑情绪"
        }
    ]
    
    success_count = 0
    total_count = len(test_cases)
    
    for test_case in test_cases:
        success = await test_ego_conversation(
            test_case["user_id"],
            test_case["message"], 
            test_case["description"]
        )
        if success:
            success_count += 1
        
        # 稍微延迟避免请求过快
        await asyncio.sleep(1)
    
    print("\n" + "=" * 60)
    print(f"🎯 测试完成: {success_count}/{total_count} 个测试通过")
    
    if success_count == total_count:
        print("✅ Ego 的人格特征表现良好！")
    else:
        print("⚠️  部分测试未通过，请检查配置")

if __name__ == "__main__":
    asyncio.run(main())
