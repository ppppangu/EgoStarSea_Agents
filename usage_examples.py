"""
HybridAgent 使用示例
"""
import asyncio
import json
from a2a.hybrid_agent import HybridAgent

async def example_1_user_chat():
    """示例1: 普通用户对话模式"""
    print("=== 示例1: 用户对话模式 ===")
    
    agent = HybridAgent(
        mode="user_chat",
        custom_templates={
            'user_chat': """你是一个有用的AI助手。

用户: {{message}}
AI助手:"""
        },
        token_stream_callback=lambda x: print(x, end='', flush=True)
    )
    
    async for event in agent.stream("你好，你能帮我解释什么是机器学习吗？"):
        pass  # 输出已通过callback处理
    
    print("\n" + "="*50 + "\n")

async def example_2_a2a_client():
    """示例2: A2A客户端模式 - 调用其他agents"""
    print("=== 示例2: A2A客户端模式 ===")
    
    agent = HybridAgent(
        mode="a2a_client",
        agent_urls=[
            "http://localhost:9999/",  # 假设的agent服务地址
            "http://localhost:8888/"
        ],
        token_stream_callback=lambda x: print(x, end='', flush=True)
    )
    
    async for event in agent.stream("帮我分析一下Python的性能优化方法"):
        # 可以根据event的不同阶段做不同处理
        stage = event.get('stage', '')
        if stage == 'agent_call':
            print(f"\n[调用Agent: {event.get('agent_name')}]")
        elif stage == 'complete':
            print("\n[A2A协作完成]")
    
    print("\n" + "="*50 + "\n")

async def example_3_a2a_server():
    """示例3: A2A服务端模式 - 处理工具调用"""
    print("=== 示例3: A2A服务端模式 ===")
    
    agent = HybridAgent(
        mode="a2a_server",
        mcp_url="https://gitmcp.io/google/A2A",
        token_stream_callback=lambda x: print(x, end='', flush=True)
    )
    
    async for event in agent.stream("请帮我搜索Python性能优化的最新文档"):
        stage = event.get('stage', '')
        if stage == 'tool_result':
            print(f"\n[工具调用结果 - 步骤{event.get('step')}]")
    
    print("\n" + "="*50 + "\n")

async def example_4_dynamic_mode_switching():
    """示例4: 动态模式切换"""
    print("=== 示例4: 动态模式切换 ===")
    
    agent = HybridAgent(mode="user_chat")
    
    # 开始是用户对话模式
    print("当前模式: user_chat")
    async for event in agent.stream("你好"):
        print(event.get('content', ''), end='')
    
    print("\n\n切换到A2A客户端模式...")
    
    # 切换到A2A客户端模式
    agent.switch_mode("a2a_client")
    agent.agent_urls = ["http://localhost:9999/"]
    
    print("当前模式: a2a_client")
    # 注意：实际使用时需要确保agent服务可用
    
    print("\n" + "="*50 + "\n")

async def example_5_custom_templates():
    """示例5: 自定义模板注入"""
    print("=== 示例5: 自定义模板 ===")
    
    # 创建专门的角色扮演模板
    role_play_template = """你是一个{{role}}。你的性格特点是{{personality}}。

当前场景：{{scenario}}
用户问题：{{message}}

请以{{role}}的身份回答："""
    
    agent = HybridAgent(
        mode="user_chat",
        custom_templates={'role_play': role_play_template}
    )
    
    # 设置角色上下文
    agent.update_context(
        role="资深Python开发专家",
        personality="严谨、耐心、喜欢用实例说明问题",
        scenario="技术咨询会议"
    )
    
    # 使用自定义模板
    agent.templates['user_chat'] = agent.templates['role_play']
    
    async for event in agent.stream("什么是装饰器？"):
        print(event.get('content', ''), end='')
    
    print("\n" + "="*50 + "\n")

async def example_6_context_management():
    """示例6: 上下文管理"""
    print("=== 示例6: 上下文管理 ===")
    
    chat_template = """对话历史：
{% for msg in chat_history %}
{{msg.role}}: {{msg.content}}
{% endfor %}

当前用户: {{message}}
AI助手:"""
    
    agent = HybridAgent(
        mode="user_chat",
        custom_templates={'user_chat': chat_template},
        agent_context={'chat_history': []}
    )
    
    # 模拟多轮对话
    messages = [
        "我想学习Python",
        "从哪里开始比较好？",
        "有推荐的书籍吗？"
    ]
    
    for i, msg in enumerate(messages):
        print(f"\n--- 第{i+1}轮对话 ---")
        print(f"用户: {msg}")
        print("AI助手: ", end='')
        
        response = ""
        async for event in agent.stream(msg):
            content = event.get('content', '')
            response += content
            print(content, end='')
        
        # 更新对话历史
        chat_history = agent.agent_context.get('chat_history', [])
        chat_history.append({'role': '用户', 'content': msg})
        chat_history.append({'role': 'AI助手', 'content': response})
        agent.update_context(chat_history=chat_history[-6:])  # 保持最近3轮对话
    
    print("\n" + "="*50 + "\n")

# HTTP API 使用示例
def api_usage_examples():
    """API使用示例"""
    print("=== API使用示例 ===")
    
    # 1. 普通用户对话
    user_chat_example = {
        "method": "POST",
        "url": "http://localhost:8000/chat",
        "json": {
            "user_id": "user123",
            "session_id": "normal_session_456", 
            "message": "你好，请介绍一下Python"
        }
    }
    
    # 2. A2A模式对话（session_id以a2a_开头）
    a2a_chat_example = {
        "method": "POST", 
        "url": "http://localhost:8000/chat",
        "json": {
            "user_id": "user123",
            "session_id": "a2a_session_789",
            "message": "帮我分析代码质量"
        }
    }
    
    # 3. 专用A2A端点
    a2a_specialized_example = {
        "method": "POST",
        "url": "http://localhost:8000/a2a/chat", 
        "json": {
            "user_id": "user123",
            "session_id": "a2a_session_999",
            "message": "执行代码分析任务",
            "agent_urls": [
                "http://code-analyzer:9999/",
                "http://security-checker:8888/"
            ],
            "mode": "a2a_client"
        }
    }
    
    examples = [
        ("普通用户对话", user_chat_example),
        ("A2A模式对话", a2a_chat_example), 
        ("专用A2A端点", a2a_specialized_example)
    ]
    
    for name, example in examples:
        print(f"\n{name}:")
        print(f"curl -X {example['method']} \\")
        print(f"  {example['url']} \\")
        print(f"  -H 'Content-Type: application/json' \\")
        print(f"  -d '{json.dumps(example['json'], ensure_ascii=False, indent=2)}'")

# 命令行使用示例
def cli_usage_examples():
    """命令行使用示例"""
    print("\n=== 命令行使用示例 ===")
    
    # 启动服务
    print("1. 启动服务:")
    print("cd C:\\Users\\meidi\\Desktop\\EgoStarSea_Agents")
    print("python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000")
    
    # 测试健康检查
    print("\n2. 测试健康检查:")
    print("curl http://localhost:8000/health")
    
    # 用户对话测试
    print("\n3. 用户对话测试:")
    print("""curl -X POST http://localhost:8000/chat \\
  -H 'Content-Type: application/json' \\
  -d '{
    "user_id": "test_user",
    "session_id": "test_session", 
    "message": "你好，介绍一下你自己"
  }'""")
    
    # A2A协议测试
    print("\n4. A2A协议测试:")
    print("""curl -X POST http://localhost:8000/a2a/chat \\
  -H 'Content-Type: application/json' \\
  -d '{
    "user_id": "test_user",
    "session_id": "a2a_test",
    "message": "执行代码分析",
    "agent_urls": ["http://localhost:9999/"],
    "mode": "a2a_client"
  }'""")

async def main():
    """运行所有示例"""
    await example_1_user_chat()
    # await example_2_a2a_client()  # 需要实际的agent服务
    # await example_3_a2a_server()  # 需要MCP服务
    await example_4_dynamic_mode_switching()
    await example_5_custom_templates()
    await example_6_context_management()
    
    api_usage_examples()
    cli_usage_examples()

if __name__ == "__main__":
    print("HybridAgent 使用示例\n")
    asyncio.run(main())