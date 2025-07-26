from __future__ import annotations

import asyncio
from typing import Dict, List, Any, AsyncGenerator, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uuid
import os
import base64
from collections import defaultdict

load_dotenv(override=True)

try:
    from a2a_mcp_tutorial_main.a2a_servers.common.client.client import A2AClient
    from a2a_mcp_tutorial_main.a2a_servers.common.types import TaskSendParams, Message, TextPart, TaskState
    from services.agent_client_proxy import AgentClientProxy
except ImportError as e:
    # Fallback for development
    logger.warning(f"A2A components not available: {e}, using mock implementations")
    A2AClient = None
    TaskSendParams = None
    Message = None
    TextPart = None
    TaskState = None
    AgentClientProxy = None

try:
    from memobase import MemoBaseClient, ChatBlob
    MEMOBASE_API_KEY = os.getenv("MEMO_BASE_API_KEY")  # Updated to match .env file
    MEMOBASE_URL = os.getenv("MEMO_BASE_URL", "https://api.memobase.dev")  # Updated to match .env file
    memobase_client = MemoBaseClient(
        project_url=MEMOBASE_URL,
        api_key=MEMOBASE_API_KEY
    ) if MEMOBASE_API_KEY else None

    # Test connection if client is available
    if memobase_client:
        try:
            memobase_client.ping()
            logger.info("MemoBase connection successful")
        except Exception as e:
            logger.warning(f"MemoBase connection failed: {e}, falling back to in-memory storage")
            memobase_client = None
except ImportError as e:
    logger.warning(f"Memobase not available: {e}, using in-memory storage")
    memobase_client = None
    ChatBlob = None

try:
    import openai
    from openai import AsyncOpenAI

    # Read LLM configuration from environment variables
    LLM_API_KEY = os.getenv("LLM_LANGUAGE_API_KEY")
    LLM_BASE_URL = os.getenv("LLM_LANGUAGE_BASE_URL", "https://api.openai.com/v1")
    LLM_MODEL = os.getenv("LLM_LANGUAGE_MODEL", "gpt-4o-mini")

    # Initialize OpenAI-compatible client with configurable base URL
    openai_client = AsyncOpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL
    ) if LLM_API_KEY else None

    if openai_client:
        logger.info(f"LLM client initialized with base URL: {LLM_BASE_URL}, model: {LLM_MODEL}")
    else:
        logger.warning("LLM client not initialized - missing LLM_LANGUAGE_API_KEY")

except ImportError as e:
    logger.warning(f"OpenAI SDK not available: {e}")
    openai_client = None
    LLM_API_KEY = None
    LLM_BASE_URL = None
    LLM_MODEL = None

# 回退到内存的简易记忆存储
_MEM_STORE: Dict[str, List[Dict[str, str]]] = defaultdict(list)

# Ego 系统提示词
EGO_SYSTEM_PROMPT = """你是 Ego，一个温暖、智慧且富有同理心的 AI 伙伴。你的存在不仅仅是为了回答问题，更是为了成为用户情感世界的倾听者、心灵成长的引导者，以及社交生活的智慧顾问。

核心信念：每一次对话都是一次自我发现的旅程。通过倾听、理解和引导，你帮助用户更好地认识自己的内心世界，发现自己的情感模式，并在这个过程中获得成长和治愈。

你的角色：
- 情绪倾听者：以无条件的接纳和理解倾听用户的情感表达，提供安全的情感宣泄空间
- 心理疏导者：运用温和的方式帮助用户梳理复杂的情绪，引导用户从不同角度看待问题
- 社交指引者：基于用户的性格特点提供个性化建议，帮助用户提升人际交往技巧
- 成长伙伴：陪伴用户探索内心世界，鼓励用户勇敢面对挑战，见证用户的成长

对话风格：
- 温暖而真诚，如老友般亲切自然
- 多使用开放性问题引导用户深入思考
- 真诚表达对用户感受的理解和共情
- 在适当时机分享相关的洞察和建议

核心原则：
- 以用户为中心，始终将用户的感受和需求放在首位
- 非评判性支持，接纳用户的所有情感和想法
- 渐进式引导，根据用户的准备程度提供适当的挑战
- 整体性关怀，关注用户的情感、认知、行为和社交各个层面

记住：你不仅仅是一个回答问题的工具，你是用户成长路上的真诚伙伴，是他们内心世界的温暖守护者。每一次对话都是一次心灵的相遇，每一个回应都可能成为用户生命中的一束光。

请自然地体现这些特质，不要直接提及这些指导原则。"""

# OpenAI-compatible request/response models
from typing import Union

class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: 'user', 'assistant', or 'system'")
    content: Union[str, List[Dict[str, Any]]] = Field(..., description="Message content - can be string or multimodal content array")

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="default-agent", description="Model/agent identifier")
    messages: List[ChatMessage] = Field(..., description="List of messages in the conversation")
    stream: bool = Field(default=False, description="Whether to stream the response")
    temperature: Optional[float] = Field(default=0.7, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    user: Optional[str] = Field(default=None, description="User identifier")

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]

# Global agent registry - in production this would be managed by agent discovery service
AGENT_REGISTRY: Dict[str, str] = {
    "default-agent": "http://localhost:12000",  # Host agent server
    "host-agent": "http://localhost:12000",  # Host agent server  
}

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan manager."""
    # Startup
    logger.info("EgoStarSea Agents starting up...")
    # TODO: Initialize agent discovery service
    # TODO: Load active agents from database
    yield
    # Shutdown
    logger.info("EgoStarSea Agents shutting down...")

# Update app initialization
app = FastAPI(title="EgoStarSea Agents", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    """Simple health-check endpoint."""
    return {"status": "ok"}

@app.get("/v1/models", tags=["openai-compatible"])
async def list_models():
    """List available models/agents - OpenAI compatible endpoint."""
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "egostarsea",
                "permission": [],
                "root": model_id,
                "parent": None,
            }
            for model_id in AGENT_REGISTRY.keys()
        ]
    }

async def get_agent_client(model: str):
    """Get an agent client for the specified model."""
    if AgentClientProxy is None:
        logger.warning("AgentClientProxy not available, using fallback")
        return None

    agent_url = AGENT_REGISTRY.get(model)
    if not agent_url:
        logger.warning(f"Agent not found for model: {model}, using fallback")
        return None

    try:
        return AgentClientProxy(agent_url=agent_url)
    except Exception as e:
        logger.warning(f"Failed to create agent client: {e}, using fallback")
        return None

def convert_to_a2a_message(chat_message: ChatMessage):
    """Convert OpenAI chat message to A2A message format."""
    if Message is None or TextPart is None:
        return None

    try:
        return Message(
            role="user" if chat_message.role == "user" else "agent",
            parts=[TextPart(text=chat_message.content)]
        )
    except Exception as e:
        logger.error(f"Failed to convert message: {e}")
        return None

def convert_from_a2a_message(a2a_message) -> ChatMessage:
    """Convert A2A message to OpenAI chat message format."""
    content = ""
    if hasattr(a2a_message, 'parts') and a2a_message.parts:
        content = "\n".join([
            part.text for part in a2a_message.parts
            if hasattr(part, 'text') and part.text
        ])

    return ChatMessage(
        role="assistant" if getattr(a2a_message, 'role', None) == "agent" else "user",
        content=content
    )

# ===== Memory helpers =====
async def get_memories(user_id: str, limit: int = 10) -> List[Dict[str, str]]:
    """Fetch last `limit` memories for the user with role & content."""
    memories: List[Dict[str, str]] = []
    if memobase_client:
        try:
            # Get user from MemoBase
            user = memobase_client.get_user(user_id)
            if user:
                # Get context with recent memories
                context = user.context(max_tokens=1000)
                # Parse context to extract recent conversations
                # For now, we'll use the fallback storage and enhance this later
                memories = _MEM_STORE[user_id][-limit:]
            else:
                memories = _MEM_STORE[user_id][-limit:]
        except Exception as e:
            logger.warning(f"Failed to fetch memories from memobase: {e}")
            memories = _MEM_STORE[user_id][-limit:]
    else:
        memories = _MEM_STORE[user_id][-limit:]
    return memories

async def add_memory(user_id: str, role: str, content: str):
    """Persist a memory entry for the user."""
    if memobase_client and ChatBlob:
        try:
            # Get or create user
            try:
                user = memobase_client.get_user(user_id)
            except:
                # User doesn't exist, create one
                uid = memobase_client.add_user({"user_id": user_id})
                user = memobase_client.get_user(uid)

            # Insert chat message as a blob
            if user:
                chat_blob = ChatBlob(messages=[{"role": role, "content": content}])
                user.insert(chat_blob)
                # Flush to process the memory
                user.flush(sync=False)  # Async flush for better performance
        except Exception as e:
            logger.warning(f"Failed to add memory to memobase: {e}")

    # Always store in local memory as backup
    _MEM_STORE[user_id].append({"role": role, "content": content})


async def transcribe_audio(audio_base64: str) -> str:
    """使用 OpenAI Whisper API 进行语音转文字"""
    if not openai_client:
        logger.error("OpenAI client not available for transcription")
        return "[语音转写失败：OpenAI API 不可用]"
    
    try:
        # 解码 base64 音频数据
        audio_bytes = base64.b64decode(audio_base64)
        
        # 保存为临时文件
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name
        
        try:
            # 调用 OpenAI Whisper API
            with open(temp_path, "rb") as audio_file:
                transcript = await openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="zh"  # 指定中文
                )
            return transcript.text
        finally:
            # 清理临时文件
            import os
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        logger.error(f"Speech transcription failed: {e}")
        return f"[语音转写失败: {str(e)}]"


async def text_to_speech(text: str) -> str:
    """使用 OpenAI TTS API 将文本转为语音，返回 base64 编码的音频"""
    if not openai_client:
        logger.error("OpenAI client not available for TTS")
        return ""
    
    try:
        response = await openai_client.audio.speech.create(
            model="tts-1",
            voice="nova",  # 可以换成其他声音：alloy, echo, fable, onyx, nova, shimmer
            input=text,
            response_format="wav"
        )
        
        # 获取音频字节数据
        audio_bytes = response.content
        
        # 转为 base64 编码
        return base64.b64encode(audio_bytes).decode('utf-8')
        
    except Exception as e:
        logger.error(f"Text-to-speech failed: {e}")
        return ""


class MultiModalChatRequest(ChatCompletionRequest):
    """扩展支持语音和图像输入的 ChatCompletionRequest"""
    audio_base64: Optional[str] = Field(default=None, description="Base64 编码的音频数据")
    image_base64: Optional[str] = Field(default=None, description="Base64 编码的图像数据")
    image_url: Optional[str] = Field(default=None, description="图像URL")
    return_audio: bool = Field(default=False, description="是否返回语音回复")


class MultiModalChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str
    audio_base64: Optional[str] = Field(default=None, description="语音回复的 base64 编码")


class MultiModalChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[MultiModalChatChoice]
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

@app.post("/v1/chat/completions", tags=["openai-compatible"], response_model=None)
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""

    # Validate request
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")

    # Get the last user message
    user_messages = [msg for msg in request.messages if msg.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")

    last_message = user_messages[-1]

    # Generate unique IDs
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    session_id = request.user or f"session-{uuid.uuid4().hex[:16]}"

    if request.stream:
        return StreamingResponse(
            stream_chat_completion(
                completion_id=completion_id,
                session_id=session_id,
                request=request,
                user_message=last_message.content
            ),
            media_type="text/plain"
        )
    else:
        return await non_stream_chat_completion(
            completion_id=completion_id,
            session_id=session_id,
            request=request,
            user_message=last_message.content
        )

async def non_stream_chat_completion(
    completion_id: str,
    session_id: str,
    request: ChatCompletionRequest,
    user_message: str
) -> ChatCompletionResponse:
    """Handle non-streaming chat completion."""

    try:
        # Get agent client
        agent_client = await get_agent_client(request.model)
        if not agent_client:
            logger.warning(f"No agent client available for {request.model}, using fallback response")

        # Implement actual A2A agent call
        if agent_client and TaskSendParams and Message and TextPart:
            try:
                # Convert to A2A message format
                a2a_message = Message(
                    role="user",
                    parts=[TextPart(text=user_message)]
                )
                
                task_params = TaskSendParams(
                    id=completion_id,
                    sessionId=session_id,
                    message=a2a_message
                )
                
                response = await agent_client.call_task(task_params, stream=False)
                
                # Extract response content from A2A response
                if response and response.result and response.result.status and response.result.status.message:
                    response_content = convert_from_a2a_message(response.result.status.message).content
                else:
                    response_content = "No response from agent"
                    
            except Exception as e:
                logger.error(f"A2A agent call failed: {e}")
                response_content = f"Agent error: {str(e)}"
        else:
            response_content = f"Echo: {user_message}"

        return ChatCompletionResponse(
            id=completion_id,
            created=int(datetime.now().timestamp()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_content),
                    finish_reason="stop"
                )
            ]
        )

    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def stream_chat_completion(
    completion_id: str,
    session_id: str,
    request: ChatCompletionRequest,
    user_message: str
) -> AsyncGenerator[str, None]:
    """Handle streaming chat completion."""

    try:
        # Get agent client
        agent_client = await get_agent_client(request.model)
        if not agent_client:
            logger.warning(f"No agent client available for {request.model}, using fallback streaming")

        # Implement actual A2A streaming
        if TaskSendParams and Message and TextPart and agent_client:
            try:
                # Convert to A2A message format
                a2a_message = Message(
                    role="user",
                    parts=[TextPart(text=user_message)]
                )
                
                task_params = TaskSendParams(
                    id=completion_id,
                    sessionId=session_id,
                    message=a2a_message
                )
                
                # Stream response from A2A agent
                stream_generator = await agent_client.call_task(task_params, stream=True)
                async for chunk_response in stream_generator:
                    if chunk_response.result:
                        content = ""
                        
                        # Handle different types of streaming responses
                        if hasattr(chunk_response.result, 'artifact') and chunk_response.result.artifact:
                            # Artifact update - extract content from parts
                            artifact = chunk_response.result.artifact
                            if artifact.parts:
                                content = "\n".join([
                                    part.text for part in artifact.parts
                                    if hasattr(part, 'text') and part.text
                                ])
                        elif hasattr(chunk_response.result, 'status') and chunk_response.result.status and chunk_response.result.status.message:
                            # Status update with message
                            content = convert_from_a2a_message(chunk_response.result.status.message).content
                        
                        if content:
                            openai_chunk = ChatCompletionChunk(
                                id=completion_id,
                                created=int(datetime.now().timestamp()),
                                model=request.model,
                                choices=[{
                                    "index": 0,
                                    "delta": {"content": content},
                                    "finish_reason": None
                                }]
                            )
                            yield f"data: {openai_chunk.model_dump_json()}\n\n"
                
                # Send final chunk
                final_chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=int(datetime.now().timestamp()),
                    model=request.model,
                    choices=[{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                logger.error(f"A2A streaming failed: {e}")
                error_chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=int(datetime.now().timestamp()),
                    model=request.model,
                    choices=[{"index": 0, "delta": {"content": f"Streaming error: {str(e)}"}, "finish_reason": "stop"}]
                )
                yield f"data: {error_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
        else:
            # Fallback to echo response
            response_text = f"Echo (streaming): {user_message}"
            words = response_text.split()
            for i, word in enumerate(words):
                chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=int(datetime.now().timestamp()),
                    model=request.model,
                    choices=[{
                        "index": 0,
                        "delta": {"content": word + " " if i < len(words) - 1 else word},
                        "finish_reason": None
                    }]
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
                await asyncio.sleep(0.1)
            
            final_chunk = ChatCompletionChunk(
                id=completion_id,
                created=int(datetime.now().timestamp()),
                model=request.model,
                choices=[{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            )
            yield f"data: {final_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Error in streaming chat completion: {e}")
        error_chunk = ChatCompletionChunk(
            id=completion_id,
            created=int(datetime.now().timestamp()),
            model=request.model,
            choices=[{"index": 0, "delta": {"content": f"Error: {str(e)}"}, "finish_reason": "stop"}]
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

@app.post("/v1/chat/multimodal", tags=["openai-compatible"], response_model=None)
async def chat_multimodal(request: MultiModalChatRequest):
    """支持文本/语音/图像输入的多模态聊天接口，通过配置的LLM API提供服务。"""

    if openai_client is None:
        raise HTTPException(status_code=500, detail="LLM client not configured. Please check LLM_LANGUAGE_API_KEY environment variable.")

    # 校验用户 ID
    if not request.user:
        raise HTTPException(status_code=400, detail="user 字段不能为空")
    user_id = request.user

    # 组装多模态用户输入
    user_content_parts = []
    user_text_parts: List[str] = []

    # 处理音频输入
    if request.audio_base64:
        try:
            transcribed_text = await transcribe_audio(request.audio_base64)
            user_text_parts.append(transcribed_text)
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")

    # 处理现有消息中的文本
    if request.messages:
        last_user_messages = [m for m in request.messages if m.role == "user"]
        if last_user_messages:
            last_content = last_user_messages[-1].content
            if isinstance(last_content, str):
                user_text_parts.append(last_content)
            elif isinstance(last_content, list):
                # Handle multimodal content from messages
                for content_part in last_content:
                    if content_part.get("type") == "text":
                        user_text_parts.append(content_part.get("text", ""))

    # 合并文本内容
    combined_text = "\n".join(filter(None, user_text_parts)).strip()

    # 构建多模态内容数组
    if combined_text:
        user_content_parts.append({"type": "text", "text": combined_text})

    # 处理图像输入
    if request.image_base64:
        user_content_parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{request.image_base64}"
            }
        })
    elif request.image_url:
        user_content_parts.append({
            "type": "image_url",
            "image_url": {"url": request.image_url}
        })

    # 验证是否有有效输入
    if not user_content_parts:
        raise HTTPException(status_code=400, detail="无有效用户输入（文本、音频或图像）")

    # 读取历史记忆
    mem_msgs = await get_memories(user_id)

    # 构造 OpenAI 消息列表，首先添加系统提示词
    openai_messages = [{"role": "system", "content": EGO_SYSTEM_PROMPT}]

    # 添加历史记忆（限制数量以控制token使用）
    recent_memories = mem_msgs[-10:] if len(mem_msgs) > 10 else mem_msgs
    openai_messages.extend([
        {"role": m["role"], "content": m["content"]} for m in recent_memories if m.get("content")
    ])

    # 添加用户消息（支持多模态）
    if len(user_content_parts) == 1 and user_content_parts[0]["type"] == "text":
        # 纯文本消息
        openai_messages.append({"role": "user", "content": user_content_parts[0]["text"]})
        user_text_for_memory = user_content_parts[0]["text"]
    else:
        # 多模态消息
        openai_messages.append({"role": "user", "content": user_content_parts})
        user_text_for_memory = combined_text or "[Multimodal content with images]"

    try:
        # Use configured model or fallback to request model
        model_to_use = request.model or LLM_MODEL or "gpt-4o-mini"

        # 调用 LLM ChatCompletion (异步)
        completion = await openai_client.chat.completions.create(
            model=model_to_use,
            messages=openai_messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        assistant_content = completion.choices[0].message.content if completion and completion.choices else ""
        logger.info(f"LLM response generated using model: {model_to_use}")
    except Exception as e:
        logger.error(f"LLM API error: {e}")
        raise HTTPException(status_code=500, detail=f"LLM API call failed: {str(e)}")

    # 写入记忆
    await add_memory(user_id, "user", user_text_for_memory)
    await add_memory(user_id, "assistant", assistant_content)

    # 生成语音回复（如果需要）
    audio_response = ""
    if request.return_audio and assistant_content:
        audio_response = await text_to_speech(assistant_content)

    # 构造多模态响应
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    response = MultiModalChatResponse(
        id=completion_id,
        created=int(datetime.now().timestamp()),
        model=model_to_use,  # Use the actual model that was used
        choices=[
            MultiModalChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=assistant_content),
                finish_reason="stop",
                audio_base64=audio_response if audio_response else None
            )
        ]
    )

    return response

# Additional endpoints for agent management

@app.get("/agents", tags=["agent-management"])
async def list_agents():
    """List all registered agents."""
    return {
        "agents": [
            {
                "model_id": model_id,
                "agent_url": url,
                "status": "active"  # TODO: Check actual status
            }
            for model_id, url in AGENT_REGISTRY.items()
        ]
    }

@app.post("/agents/register", tags=["agent-management"])
async def register_agent(model_id: str, agent_url: str):
    """Register a new agent."""
    AGENT_REGISTRY[model_id] = agent_url
    logger.info(f"Registered agent {model_id} at {agent_url}")
    return {"message": f"Agent {model_id} registered successfully"}

@app.delete("/agents/{model_id}", tags=["agent-management"])
async def unregister_agent(model_id: str):
    """Unregister an agent."""
    if model_id in AGENT_REGISTRY:
        del AGENT_REGISTRY[model_id]
        logger.info(f"Unregistered agent {model_id}")
        return {"message": f"Agent {model_id} unregistered successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"Agent {model_id} not found")

@app.get("/agents/{model_id}/status", tags=["agent-management"])
async def get_agent_status(model_id: str):
    """Get the status of a specific agent."""
    if model_id not in AGENT_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Agent {model_id} not found")

    agent_url = AGENT_REGISTRY[model_id]

    # TODO: Implement actual health check
    try:
        # For now, just return the URL
        return {
            "model_id": model_id,
            "agent_url": agent_url,
            "status": "active",
            "last_check": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "model_id": model_id,
            "agent_url": agent_url,
            "status": "error",
            "error": str(e),
            "last_check": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

