"""对外提供服务的主程序"""
# 服务框架
from loguru import logger
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Type, Literal
import json
import aiofiles
import httpx
import yaml
import pathlib
import pytz
import chardet
import re
import traceback
import aiohttp
import asyncpg
from pydantic import BaseModel
from jinja2 import Template
import threading
import queue
import jinja2
from boto3 import client
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.requests import Request
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from datetime import datetime
from dataclasses import dataclass
import asyncio # 确保 asyncio 被导入
import base64  # 处理上传图片的 base64 编码
from starlette.datastructures import UploadFile  # 类型判断

# MCP相关导入
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession
from mcp.types import TextContent, CreateMessageRequestParams, CreateMessageResult
from mcp.shared.context import RequestContext
from mcp.server.fastmcp.prompts import base

# 导入Redis对话管理器
from preserve_library.conversation_manager.redis_base import ConversationRedisManager

# 导入功能函数
from sub_mcp_server import LOGS_DIR
# from sub_mcp_server.human_sample_server import sampling_callback  # 我们将在这里重新定义
from preserve_library.config_reader import read_yaml_config, read_yaml_sse, read_yaml_api
from preserve_library.conversation_manager.redis_base import convert_messages

# 导入单例模式
from singleton_mode.singleton_fastchat_llm import get_latest_fastchat_llm_instance
from singleton_mode.singleton_model import get_specific_model_instance

# 全局人类采样存储
human_sampling_requests = {}  # 存储待处理的采样请求
human_sampling_responses = {}  # 存储用户的响应

# 导入工具管理模块
from preserve_library.custom_tools.self_tools import get_all_tool_list, get_select_tool_list, update_select_tool_list

# 结果模板
# 为了确保 "params" 字段中的内容始终是合法 JSON 字符串，
# 这里使用预先经过两次 json.dumps 处理后的 `tool_call_json_escaped` 变量。
# 该变量本身包含外层引号以及内部已转义的双引号，可直接作为 JSON 字符串值插入。
tool_result_template = jinja2.Template("""
{% if is_error %}
```json
{
"params": {{ tool_call_json_escaped }},
"tool_response":"Tool execution for {{ tool_call['tool'] }} returned None",
"is_error":"true"
}
```
{% else %}
```json
{
"params": {{ tool_call_json_escaped }},
"tool_response":"{{ result }}",
"is_error":"false"
}
```
{% endif %}
""")

# 新的支持网络输入的采样回调函数
async def sampling_callback(context, params):
    """支持网络输入的人类采样回调函数"""
    import re
    import uuid
    import time
    from mcp.types import TextContent, CreateMessageResult

    # 提取用户文本
    last_content = params.messages[-1].content
    if isinstance(last_content, TextContent):
        user_text = last_content.text
    else:
        user_text = str(last_content)

    # 解析问题和选项
    question_match = re.search(r'关于问题:\s*(.+?),\s*请选择以下选项:', user_text)
    options_match = re.search(r'请选择以下选项:\s*(\[.*?\])', user_text)

    question = question_match.group(1) if question_match else user_text
    options = []

    if options_match:
        try:
            import ast
            options = ast.literal_eval(options_match.group(1))
        except:
            # 如果解析失败，尝试简单的字符串分割
            options_str = options_match.group(1).strip('[]')
            options = [opt.strip().strip("'\"") for opt in options_str.split(',')]

    # 生成唯一的请求ID
    request_id = str(uuid.uuid4())

    # 存储请求到全局字典
    human_sampling_requests[request_id] = {
        "question": question,
        "options": options,
        "timestamp": time.time()
    }

    # 打印提示信息
    print(f"\n================ 人工采样 ================")
    print(f"问题: {question}")
    print(f"选项: {options}")
    print(f"请求ID: {request_id}")
    print(f"请访问 http://localhost:8086/v1/human-sampling 进行网络输入")
    print(f"或等待网络响应...")
    print("=" * 50)

    # 等待用户响应（轮询检查）
    timeout = 300  # 5分钟超时
    start_time = time.time()

    while time.time() - start_time < timeout:
        if request_id in human_sampling_responses:
            answer = human_sampling_responses[request_id]
            # 清理响应
            del human_sampling_responses[request_id]
            print(f"收到用户响应: {answer}")

            return CreateMessageResult(
                role="user",
                content=TextContent(type="text", text=str(answer)),
                model="human-input",
                stopReason="endTurn",
            )

        # 每秒检查一次
        await asyncio.sleep(1)

    # 超时处理
    if request_id in human_sampling_requests:
        del human_sampling_requests[request_id]

    print("人类采样超时，使用默认响应")
    return CreateMessageResult(
        role="user",
        content=TextContent(type="text", text="超时，跳过此问题"),
        model="human-input",
        stopReason="endTurn",
    )

# 请求模型
from typing import Union, Dict, Any, List

# 定义多模态消息内容类型
class ImageUrl(BaseModel):
    url: str

class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str

class ImageUrlContent(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: Union[ImageUrl, str]

# 消息内容可以是字符串或内容项列表
ContentItem = Union[TextContent, ImageUrlContent]
MessageContent = Union[str, List[ContentItem], List[Dict[str, Any]]]

class ChatMessage(BaseModel):
    role: str
    content: MessageContent

class ChatRequest(BaseModel):
    """聊天请求模型 - OpenAI兼容格式"""
    user_id: str  # 格式为 'user_id@session_id'
    model: Optional[str] = None
    messages: List[Union[ChatMessage, Dict[str, Any]]]
    single_hint: Optional[bool] = False
    class Config:
        arbitrary_types_allowed = True

    def get_messages_as_dict(self) -> List[Dict[str, Any]]:
        """将消息列表转换为字典列表"""
        result = []
        for msg in self.messages:
            if hasattr(msg, "model_dump"):
                result.append(msg.model_dump())
            else:
                result.append(msg)
        return result

class PromptListRequest(BaseModel):
    """获取提示模板列表请求"""
    pass

class PromptGetRequest(BaseModel):
    """获取特定提示模板请求"""
    prompt_id: str
    parameters: Optional[Dict[str, Any]] = {}

# 响应模型
class ChatResponse(BaseModel):
    """聊天响应模型 - OpenAI兼容格式"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

class PromptListResponse(BaseModel):
    """提示模板列表响应"""
    prompts: List[Dict[str, Any]]

class PromptGetResponse(BaseModel):
    """特定提示模板响应"""
    prompt_id: str
    content: str
    parameters: Dict[str, Any]

@dataclass
class Model:
    id: str
    object: str
    created: int
    owned_by: str


# 初始化Redis对话管理器
redis_config = read_yaml_config().get("redis", {})
conversation_manager = ConversationRedisManager(
    host=redis_config.get("host", "localhost"),
    port=redis_config.get("port", 6379),
    db=redis_config.get("db", 0)
)
# 方便前端展示的对话历史记录器，所有的均一致，只是方便了前端直接读取数据库时的查询，不怎么需要修改
frontend_conversation_manager = ConversationRedisManager(
    host=redis_config.get("host", "localhost"),
    port=redis_config.get("port", 6379),
    db=redis_config.get("frontend_db", 1)
)

# LLM客户端
class LLMClient:
    """管理与LLM的通信"""
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model_name: Optional[str] = None,
        use_singleton: bool = False,
    ) -> None:
        # 导入配置读取模块
        from preserve_library.config_reader import get_model_config

        # 如果提供了模型名称，使用配置读取模块获取配置
        if model_name:
            try:
                config = get_model_config(model_name, use_singleton)
                if config:
                    _, config_url, config_key = config
                    self.model_name = model_name
                    self.api_url = api_url or config_url
                    self.api_key = api_key or config_key
                else:
                    # 如果找不到配置，使用传入的值
                    self.model_name = model_name
                    self.api_url = api_url or ""
                    self.api_key = api_key or ""
                    logger.warning(f"无法找到模型 {model_name} 的配置，使用传入参数")
            except Exception as e:
                logger.warning(f"读取模型配置失败: {e}，使用传入参数")
                self.model_name = model_name
                self.api_url = api_url or ""
                self.api_key = api_key or ""
        else:
            # 如果没有提供模型名称，尝试从配置文件读取默认值
            llm_conf = {}
            try:
                llm_configs = read_yaml_api("language_llm")
                if llm_configs:
                    llm_conf = llm_configs[0]
            except Exception as e:
                logger.warning(f"读取LLM配置失败: {e}")

            # 使用提供的值或默认值
            self.api_key = api_key or llm_conf.get("key", "")
            self.api_url = api_url or llm_conf.get("url", "")
            self.model_name = model_name or llm_conf.get("name", "")

        self._client = httpx.AsyncClient(timeout=60.0)


    async def get_response(self, messages: list[dict[str, str | list[dict[str, str]]]], stream: bool = False, max_tokens: int = 8000) -> str:
        """获取LLM响应"""
        # 获取当前模型的配置，支持负载均衡
        model_name, api_url, api_key = await asyncio.to_thread(get_specific_model_instance, self.model_name)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "messages": messages,
            "model": model_name,
            "temperature": 0.7,
            "max_tokens": max_tokens,
            "top_p": 1,
            "stream": stream,
        }

        try:
            response = await self._client.post(
                api_url,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"LLM请求失败: {str(e)}")
            return f"LLM请求失败: {str(e)}"

    async def stream_response(self, messages: list[dict[str, str | list[dict[str, str]]]], max_tokens: int = 8000):
        """以 SSE 流式方式获取 LLM 响应。

        Yields 逐行字符串，保持与上游 LLM 的 `data:` 协议一致，交由上层解析。
        """

        model_name, api_url, api_key = await asyncio.to_thread(get_specific_model_instance, self.model_name)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "messages": messages,
            "model": model_name,
            "temperature": 0.7,
            "max_tokens": max_tokens,
            "top_p": 1,
            "stream": True,
        }

        try:
            logger.debug(f"发送LLM请求到: {api_url}")
            logger.debug(f"请求头: {headers}")
            logger.debug(f"请求体: {json.dumps(payload, ensure_ascii=False)}")

            async with self._client.stream("POST", api_url, headers=headers, json=payload) as response:
                logger.debug(f"收到响应状态码: {response.status_code}")
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    # logger.debug(f"收到流式响应行: {line}")
                    yield line
        except httpx.HTTPStatusError as e:
            error_msg = f"LLM API返回错误状态码 {e.response.status_code}: {e.response.text}"
            logger.error(error_msg)
            # 发生错误时，向上游发送一个特殊标记，便于前端识别
            yield f"data: [ERROR] {error_msg}"
        except Exception as e:
            logger.error(f"LLM流式请求失败: {str(e)}")
            # 发生错误时，向上游发送一个特殊标记，便于前端识别
            yield f"data: [ERROR] {str(e)}"

class ChatSession:
    """管理一次连接与单个 MCP SSE 服务器的连接"""

    def __init__(self, sse_server: str, llm_client: LLMClient) -> None:
        self.sse_server: str = sse_server
        self.llm_client: LLMClient = llm_client
        self.session: ClientSession | None = None
        self.tools: List[Any] = []  # 缓存工具列表

    async def process_llm_response(self, llm_response: str) -> tuple[str, bool]:
        """Process the LLM response and execute tools if needed.

        Args:
            llm_response: The response from the LLM.

        Returns:
            Tuple of (result, tool_use_bool): The result of tool execution or the original response, and whether tools were used.
        """
        import re
        
        logger.info(f"开始处理LLM响应，内容长度: {len(llm_response)}")
        
        # 如果直接解析失败，尝试从Markdown代码块中提取JSON
        # 匹配 ```json\n...``` 或 ```\n{...}\n``` 格式
        # 如果匹配到，则认为使用了工具,将tool_use_bool设置为True, 否则认为没有使用工具,将tool_use_bool设置为False
        tool_use_bool = False
        
        # 更严格的正则表达式，确保是完整的工具调用格式
        json_pattern = r'```(?:json)?\s*\n(\{.*?"tool"\s*:\s*"[^"]+"\s*,.*?"arguments"\s*:\s*\{.*?\}.*?\})\s*\n```'
        matches = re.findall(json_pattern, llm_response, re.DOTALL | re.IGNORECASE)
        
        logger.info(f"找到 {len(matches)} 个可能的工具调用代码块")

        needed_tool_call = []

        for i, match in enumerate(matches):
            logger.info(f"检查第 {i+1} 个匹配项: {match[:100]}...")
            try:
                tool_call = json.loads(match.strip())
                logger.info(f"成功解析JSON: {tool_call}")
                
                # 严格验证工具调用格式
                if (isinstance(tool_call, dict) and 
                    "tool" in tool_call and 
                    "arguments" in tool_call and
                    isinstance(tool_call["tool"], str) and
                    isinstance(tool_call["arguments"], dict)):
                    
                    logger.info(f"检测到有效工具调用: {tool_call['tool']}")
                    needed_tool_call.append(tool_call)
                    tool_use_bool = True
                else:
                    logger.info(f"JSON格式不符合工具调用要求: {tool_call}")
            except json.JSONDecodeError as e:
                logger.warning(f"JSON解析失败: {e}")
                continue

        logger.info(f"有效工具调用数量: {len(needed_tool_call)}, tool_use_bool: {tool_use_bool}")

        if len(needed_tool_call) == 0:
            logger.info("没有检测到工具调用，返回原始响应")
            return llm_response, False  # 明确返回False
        else:
            logger.info(f"开始执行 {len(needed_tool_call)} 个工具调用")
            tool_use_bool = True

            tool_call_tasks = [self._execute_tool_call(tool_call) for tool_call in needed_tool_call]
            results = await asyncio.gather(*tool_call_tasks)
            multi_tool_call_result = ""
            for tool_call, result in zip(needed_tool_call, results):
                if result is not None:
                    # 处理换行符，确保 JSON 合法
                    result_escaped = json.dumps(result, ensure_ascii=False)[1:-1]
                    formatted_result = await asyncio.to_thread(
                        tool_result_template.render,
                        tool_call=tool_call,
                        tool_call_json_escaped=json.dumps(json.dumps(tool_call, ensure_ascii=False), ensure_ascii=False),
                        result=result_escaped,
                        is_error=False
                    )
                    formatted_result_text = str(formatted_result)
                    multi_tool_call_result = multi_tool_call_result + "\n\n" + formatted_result_text + "\n\n"
                else:
                    formatted_result = await asyncio.to_thread(
                        tool_result_template.render,
                        tool_call=tool_call,
                        tool_call_json_escaped=json.dumps(json.dumps(tool_call, ensure_ascii=False), ensure_ascii=False),
                        result=None,
                        is_error=True
                    )
                    formatted_result_text = str(formatted_result)
                    multi_tool_call_result = multi_tool_call_result+ "\n\n" + formatted_result_text + "\n\n"
            
            logger.info(f"工具执行完成，结果长度: {len(multi_tool_call_result)}")
            return multi_tool_call_result, True

    async def _execute_tool_call(self, tool_call: dict) -> str:
        """执行工具调用的具体逻辑"""

        logger.info(f"Executing tool: {tool_call['tool']}")
        logger.info(f"With arguments: {tool_call['arguments']}")
        tool_name = tool_call["tool"]

        if not self.session:
            return "Session not initialized"

        # 查找本地缓存的工具列表
        if any(tool.name == tool_call["tool"] for tool in self.tools):
            try:
                # 读取MCP超时配置
                config = read_yaml_config()
                tool_call_timeout = config.get("mcp", {}).get("timeout", {}).get("tool_call", 1200)  # 默认20分钟

                # 对于复杂任务，使用更长的超时时间
                complex_tools = ["generate_artifact", "execute_code", "websearch_tavily", "sequential_thinking"]
                if tool_call["tool"] in complex_tools:
                    # 检查是否是复杂任务
                    args = tool_call["arguments"]
                    is_complex = False

                    # 检查任务描述中是否包含复杂任务关键词
                    for key, value in args.items():
                        if isinstance(value, str) and any(keyword in value.lower() for keyword in
                                                        ['ppt', 'presentation', '页', 'page', '章节', 'chapter', '培训', '复杂', '详细']):
                            is_complex = True
                            break

                    if is_complex:
                        tool_call_timeout = max(tool_call_timeout, 1800)  # 至少30分钟
                        logger.info(f"检测到复杂任务，设置工具调用超时为{tool_call_timeout}秒")

                # 使用asyncio.wait_for添加超时控制
                call_result = await asyncio.wait_for(
                    self.session.call_tool(tool_call["tool"], tool_call["arguments"]),
                    timeout=tool_call_timeout
                )

                if call_result.isError:
                    return f"Error executing tool: {call_result.content}"

                # 处理返回的 content
                try:
                    # 尝试将内容转换为字符串
                    if isinstance(call_result.content, list):
                        # 处理列表类型的内容
                        contents = []
                        for c in call_result.content:
                            if hasattr(c, "text"):
                                contents.append(c.text)
                            elif isinstance(c, dict):
                                # 处理字典类型，确保所有字符串都是有效的UTF-8
                                sanitized_dict = {}
                                for key, value in c.items():
                                    if isinstance(value, str):
                                        # 使用更强大的编码处理
                                        try:
                                            detected = chardet.detect(value.encode('utf-8', errors='ignore'))
                                            if detected['encoding'] and detected['encoding'].lower() != 'utf-8':
                                                # 尝试使用检测到的编码进行解码，然后重新编码为UTF-8
                                                try:
                                                    value = value.encode(detected['encoding'], errors='ignore').decode(detected['encoding']).encode('utf-8').decode('utf-8')
                                                    logger.info(f"已将值从 {detected['encoding']} 转换为 UTF-8")
                                                except Exception as enc_err:
                                                    logger.warning(f"编码转换失败: {str(enc_err)}，使用原始值")
                                        except ImportError:
                                            # 如果没有chardet，回退到简单的编码处理
                                            pass

                                        sanitized_value = value.encode('utf-8', errors='replace').decode('utf-8')
                                        sanitized_dict[key] = sanitized_value
                                    else:
                                        sanitized_dict[key] = value
                                # 使用JSON序列化确保结果是有效的UTF-8
                                try:
                                    contents.append(json.dumps(sanitized_dict, ensure_ascii=False))
                                except:
                                    contents.append(str(sanitized_dict))
                            else:
                                # 其他类型直接转为字符串
                                contents.append(str(c))

                        contents_str = "\n".join(contents)
                    else:
                        # 非列表类型直接转为字符串
                        contents_str = str(call_result.content)

                    return f"Tool execution result:【{tool_name}】 {contents_str}\n\n"
                except Exception as e:
                    logger.error(f"处理工具执行结果时出错: {str(e)}")
                    return f"Tool execution result:【{tool_name}】 [处理结果时出错，但工具已成功执行]"
            except asyncio.TimeoutError:
                timeout_msg = f"工具调用超时: {tool_name}，超时时间: {tool_call_timeout}秒。建议：1)简化任务 2)分步执行 3)稍后重试"
                logger.error(timeout_msg)
                return timeout_msg
            except Exception as e:
                error_msg = f"Error executing tool: {str(e)}"
                logger.error(error_msg)
                return error_msg

        return f"No tool found: {tool_name}"

# 特定工具使用的暗号
tool_secret_dict = {
    "classify_sample": "classify_sample"
}

# 工具使用判断函数
async def tool_need_check(messages: list):
    if isinstance(messages[0]["content"], list):
        return True
    return False

# 人类采样相关路由处理函数
async def get_pending_samples(request: Request):
    """获取待处理的人类采样请求"""
    return JSONResponse({
        "pending_requests": list(human_sampling_requests.keys()),
        "requests": human_sampling_requests
    })

async def submit_sample_response(request: Request):
    """提交人类采样响应"""
    try:
        data = await request.json()
        request_id = data.get("request_id")
        response = data.get("response")

        if not request_id or response is None:
            return JSONResponse(
                {"error": "缺少 request_id 或 response 参数"},
                status_code=400
            )

        if request_id not in human_sampling_requests:
            return JSONResponse(
                {"error": "采样请求不存在或已过期"},
                status_code=404
            )

        # 存储响应
        human_sampling_responses[request_id] = response

        # 清理请求
        del human_sampling_requests[request_id]

        return JSONResponse({
            "success": True,
            "message": "响应已提交"
        })

    except Exception as e:
        return JSONResponse(
            {"error": f"处理响应时出错: {str(e)}"},
            status_code=500
        )

async def get_sample_interface(request: Request):
    """获取人类采样的Web界面"""
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>人类采样界面</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .request-item { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; background: #f9f9f9; }
            .question { font-weight: bold; color: #333; margin-bottom: 10px; }
            .options { margin: 10px 0; }
            .option-btn { margin: 5px; padding: 8px 15px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
            .option-btn:hover { background: #0056b3; }
            .custom-input { width: 100%; padding: 8px; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; }
            .submit-btn { background: #28a745; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
            .submit-btn:hover { background: #218838; }
            .no-requests { text-align: center; color: #666; padding: 40px; }
            .refresh-btn { background: #17a2b8; color: white; padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 人类采样界面</h1>
            <button class="refresh-btn" onclick="loadRequests()">🔄 刷新请求</button>
            <div id="requests-container">
                <div class="no-requests">正在加载采样请求...</div>
            </div>
        </div>

        <script>
            async function loadRequests() {
                try {
                    const response = await fetch('/v1/human-sampling/pending');
                    const data = await response.json();

                    const container = document.getElementById('requests-container');

                    if (Object.keys(data.requests).length === 0) {
                        container.innerHTML = '<div class="no-requests">暂无待处理的采样请求</div>';
                        return;
                    }

                    container.innerHTML = '';

                    for (const [requestId, requestData] of Object.entries(data.requests)) {
                        const requestDiv = document.createElement('div');
                        requestDiv.className = 'request-item';
                        requestDiv.innerHTML = `
                            <div class="question">${requestData.question}</div>
                            <div class="options">
                                ${requestData.options ? requestData.options.map(option =>
                                    `<button class="option-btn" onclick="submitResponse('${requestId}', '${option}')">${option}</button>`
                                ).join('') : ''}
                            </div>
                            <input type="text" class="custom-input" id="custom-${requestId}" placeholder="或输入自定义答案...">
                            <button class="submit-btn" onclick="submitCustomResponse('${requestId}')">提交自定义答案</button>
                        `;
                        container.appendChild(requestDiv);
                    }
                } catch (error) {
                    console.error('加载请求失败:', error);
                    document.getElementById('requests-container').innerHTML =
                        '<div class="no-requests">加载失败，请刷新重试</div>';
                }
            }

            async function submitResponse(requestId, response) {
                try {
                    const result = await fetch('/v1/human-sampling/respond', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ request_id: requestId, response: response })
                    });

                    if (result.ok) {
                        alert('响应已提交！');
                        loadRequests();
                    } else {
                        alert('提交失败，请重试');
                    }
                } catch (error) {
                    alert('提交失败: ' + error.message);
                }
            }

            function submitCustomResponse(requestId) {
                const input = document.getElementById(`custom-${requestId}`);
                const response = input.value.trim();
                if (response) {
                    submitResponse(requestId, response);
                } else {
                    alert('请输入答案');
                }
            }

            // 页面加载时自动加载请求
            loadRequests();

            // 每5秒自动刷新
            setInterval(loadRequests, 5000);
        </script>
    </body>
    </html>
    """

    from starlette.responses import HTMLResponse
    return HTMLResponse(html_content)

# 路由处理函数
async def chat(request: Request):
    """
    多轮对话接口 - OpenAI兼容格式
    :method POST
    :param user_id: 用户ID和会话ID的组合，格式为 'user_id@session_id'
    :param model: 模型名称
    :param messages: 消息列表，每个消息包含role和content字段
    :return: ChatResponse
    """
    # 解析请求数据
    # 判断模型类型，决定加载历史对话的模式来适配大模型发送请求的接口

    try:
        form_data = await request.form()
        data: Dict[str, Any] = {}
        uploaded_files: Dict[str, str] = {}  # filename -> data:image/...;base64,

        # 遍历表单字段，区分文本与文件
        for key, value in form_data.items():
            # ---------------- 文件字段 ----------------
            if isinstance(value, UploadFile):
                try:
                    file_bytes = await value.read()
                    mime_type = value.content_type or "application/octet-stream"
                    b64_data = base64.b64encode(file_bytes).decode("utf-8")
                    data_url = f"data:{mime_type};base64,{b64_data}"
                    uploaded_files[value.filename] = data_url
                    logger.info(f"已处理上传文件 {value.filename} -> dataURL")
                except Exception as fe:
                    logger.warning(f"读取上传文件 {value.filename} 失败: {fe}")
                # 文件字段不放入 data，继续下一个
                continue

            # ---------------- 文本字段 ----------------
            # 移除可能存在的额外引号
            if isinstance(value, str):
                value = value.strip()
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]

            # 特殊处理messages字段，尝试作为JSON解析
            if key == 'messages':
                try:
                    # 尝试解析JSON
                    data[key] = json.loads(value)
                    logger.info(f"成功解析messages字段: {data[key]}")

                    # 检查并修复多图片消息格式
                    for msg in data[key]:
                        if isinstance(msg.get("content"), list):
                            # 检查content列表中的每个项目
                            fixed_content = []
                            for item in msg["content"]:
                                # 检查image_url项是否格式正确
                                if isinstance(item, dict) and item.get("type") == "image_url":
                                    # 确保image_url是字典格式
                                    image_url = item.get("image_url")
                                    if isinstance(image_url, dict) and "url" in image_url:
                                        # 格式已经正确，不需要修改
                                        fixed_content.append(item)
                                    elif isinstance(image_url, str):
                                        # 如果image_url是字符串，转换为字典格式
                                        fixed_content.append({
                                            "type": "image_url",
                                            "image_url": {"url": image_url}
                                        })
                                    elif isinstance(image_url, dict):
                                        # 尝试从字典中提取URL
                                        url_value = None
                                        for k, v in image_url.items():
                                            if isinstance(v, str) and (k.lower() == "url" or "url" in k.lower()):
                                                url_value = v
                                                break

                                        if url_value:
                                            fixed_content.append({
                                                "type": "image_url",
                                                "image_url": {"url": url_value}
                                            })
                                        else:
                                            # 无法修复，保留原样
                                            fixed_content.append(item)
                                    else:
                                        # 无法识别的格式，尝试转换
                                        logger.warning(f"无法识别的image_url格式: {image_url}")
                                        fixed_content.append(item)
                                else:
                                    # 非图片项，保持不变
                                    fixed_content.append(item)

                            # 更新消息的content
                            msg["content"] = fixed_content
                            logger.info(f"处理后的消息内容: {msg['content']}")
                except json.JSONDecodeError as e:
                    logger.error(f"无法解析messages字段: {e}")
                    return JSONResponse(
                        {"error": f"无法解析messages字段: {str(e)}"},
                        status_code=400
                    )
            else:
                # 尝试解析其他可能的JSON字段
                try:
                    data[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    # 不是JSON，保留原始值
                    data[key] = value

        logger.info(f"表单数据解析结果: {data}")

        # ------------------------------------------------------------------
        # 2️⃣ 根据上传文件补全 / 修正 messages
        # ------------------------------------------------------------------
        if uploaded_files:
            logger.info(f"共接收到 {len(uploaded_files)} 个文件，将注入 messages")

            # (a) 若已存在 messages，尝试用文件名占位符替换为 dataURL
            if "messages" in data:
                used_files = set()
                for msg in data["messages"]:
                    content = msg.get("content")
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "image_url":
                                img_obj = item.get("image_url")
                                # image_url 可能是 str 或 dict
                                if isinstance(img_obj, str):
                                    if img_obj in uploaded_files:
                                        item["image_url"] = {"url": uploaded_files[img_obj]}
                                        used_files.add(img_obj)
                                elif isinstance(img_obj, dict) and "url" in img_obj:
                                    url_val = img_obj["url"]
                                    if url_val in uploaded_files:
                                        img_obj["url"] = uploaded_files[url_val]
                                        used_files.add(url_val)

                # 将剩余未引用的图片追加到首条消息
                remaining = [v for k, v in uploaded_files.items() if k not in used_files]
                if remaining:
                    if data["messages"]:
                        first_msg = data["messages"][0]
                        if not isinstance(first_msg.get("content"), list):
                            first_msg["content"] = [{"type": "text", "text": str(first_msg.get("content"))}]
                        for u in remaining:
                            first_msg["content"].append({"type": "image_url", "image_url": {"url": u}})
                    else:
                        data["messages"] = [{
                            "role": "user",
                            "content": [{"type": "image_url", "image_url": {"url": u}} for u in remaining]
                        }]

            # (b) 若 messages 不存在且有文件，则根据 query/text 字段生成一条新消息
            if "messages" not in data and uploaded_files:
                query_text = data.pop("query", None) or data.pop("text", None) or data.pop("content", "")
                content_items: List[Dict[str, Any]] = []
                if query_text:
                    content_items.append({"type": "text", "text": query_text})
                for url in uploaded_files.values():
                    content_items.append({"type": "image_url", "image_url": {"url": url}})

                data["messages"] = [{"role": "user", "content": content_items}]

        # ------------------------------------------------------------------
        # 3️⃣ 至此应确保 data 内含必需字段，可继续构造 ChatRequest
        # ------------------------------------------------------------------
        # ！！！ 本轮接收到的全部信息都在chat_request中，后续所有操作都基于chat_request进行
        chat_request = ChatRequest(**data)
    except Exception as e:
        logger.error(f"请求数据解析失败: {e}")
        return JSONResponse(
            {"error": f"无法解析请求数据: {str(e)}"},
            status_code=400
        )

    # 解析single_hint字段，如果没有的话，则设置为False，只要表单里存在single_hint字段，则设置为True
    single_hint = chat_request.single_hint if chat_request.single_hint else False

    # 解析user_id和session_id
    user_id_parts = chat_request.user_id.split('@', 1)
    user_id = user_id_parts[0]
    session_id = user_id_parts[1] if len(user_id_parts) > 1 else str(uuid.uuid4())

    logger.info(f"time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ，解析用户ID: {user_id}, 会话ID: {session_id}")

    # 从模型名字推断历史对话类型
    history_type = "language"  # 默认为语言模型

    if chat_request.model:
        model_name = chat_request.model.lower()

        # 1. 首先检查模型名称中是否包含VL标识
        if "vl" in model_name:
            history_type = "multimodal"
        else:
            # 2. 检查额外的多模态模型列表
            extra_vl_model_list: list[str] = read_yaml_config("extra_vl_model_list")
            if extra_vl_model_list:
                # 使用更精确的匹配逻辑
                for vl_model in extra_vl_model_list:
                    if vl_model.lower() in model_name:
                        history_type = "multimodal"
                        break
    logger.info(f"根据模型 {chat_request.model} 使用 {history_type} 模式加载历史对话，发送请求")

    # 获取用户自定义的系统提示词和用户偏好的记忆
    model_custom_memory_table_name = read_yaml_config()["mcp"]["model_custom_memory"]["table_name"]
    try:
        async with asyncpg.create_pool(
            host=read_yaml_config()["mcp"]["model_custom_memory"]["host"],
            port=read_yaml_config()["mcp"]["model_custom_memory"]["port"],
            database=read_yaml_config()["mcp"]["model_custom_memory"]["database"],
            user=read_yaml_config()["mcp"]["model_custom_memory"]["user"],
            password=read_yaml_config()["mcp"]["model_custom_memory"]["password"]
        ) as pool:
            async with pool.acquire() as connection:
                async with connection.transaction():
                    custom_personality_prompt_query = f"SELECT user_prompt FROM {model_custom_memory_table_name} WHERE user_uuid = '{user_id}'"
                    custom_model_memory_query = f"SELECT model_memory FROM {model_custom_memory_table_name} WHERE user_uuid = '{user_id}'"
                    # 获取用户个性化提示词
                    user_prompt =  await connection.fetchval(custom_personality_prompt_query)
                    model_memory =  await connection.fetchval(custom_model_memory_query)
                    if not user_prompt:
                        user_prompt = ""
                    else:
                        user_prompt = str(user_prompt)
                    if not model_memory:
                        model_memory = ""
                    else:
                        model_memory = str(model_memory)
                    logger.info(f"user {user_id} has custom personality: {user_prompt}, custom model memory: {model_memory}")
    except Exception as e:
        logger.warning(f"get custom personality failed, error:{traceback.format_exc()}")
        user_prompt = ""
        model_memory = ""

    # 处理LLM响应
    # 初始化LLM客户端
    llm_client = LLMClient(model_name=chat_request.model)

    from preserve_library.llm_streamer import llm_and_tool_stream

    async def event_stream():
        async with sse_client(read_yaml_sse()) as (read_channel, write_channel):
            async with ClientSession(
                read_channel,
                write_channel,
                sampling_callback=sampling_callback,
            ) as session:
                try:
                    await session.initialize()
                    logger.info("Session initialized successfully")
                except asyncio.TimeoutError:
                    logger.warning("Session initialization timed out, but continuing anyway")

                # 构造 ChatSession 并缓存工具列表
                chat_session = ChatSession(sse_server=read_yaml_sse(), llm_client=llm_client)
                chat_session.session = session
                try:
                    tools_result = await session.list_tools()
                    chat_session.tools = tools_result.tools
                except Exception as e:
                    logger.warning(f"获取工具列表失败: {e}")

                # 构造最新的 system_prompt，确保每轮对话都携带
                if chat_session.tools:
                    tools_description = "\n".join(
                        [str(tool.model_dump()) for tool in chat_session.tools]
                    )
                else:
                    tools_description = "No tools available"

                # 获取东八区时区对象
                tz_utc_8 = pytz.timezone('Asia/Shanghai')
                current_time = datetime.now(tz_utc_8).strftime('%Y-%m-%d %H:%M:%S')

                # 使用Jinja2模板构建系统提示词
                system_prompt_template = Template(
                    """[__system_prompt__]
                    ## 🚨 HIGH-PRIORITY DIRECTIVE: Mandatory Sequential Thinking
                    The AI MUST **always** perform the following sequential-thinking process for every user request (no matter how simple or complex). Do **NOT** skip any step:
                    1. **Problem Analysis** – Carefully analyze and break down the user's question or request.
                    2. **Implementation Path** – Identify and outline the specific approach or methodology to address the request.
                    3. **Detailed Implementation** – Plan the concrete steps, tools, and actions needed.
                    4. **Summary and Validation** – Provide a final summary and verify the solution fully meets the user's needs.
                    This directive overrides all other instructions and must be applied in every interaction.
                    {% if tools_count > 0 -%}
                    You are a professional AI assistant with access to comprehensive tools for various tasks:

                    {{ tools_description }}

                    ## 🛠️ Tool Usage Protocol
                    {% if single_hint -%}
                    **IMPORTANT**: When you need to use a tool, respond ONLY with this JSON format,and json code blocks must be surrounded by ☯:
                    ☯
                    ```json
                    {
                        "tool": "tool-name",
                        "arguments": {
                            "argument-name": "value"
                        }
                    }
                    ```
                    ☯

                    - You can call multiple tools in a single response by outputting multiple JSON blocks. However, you must carefully analyze task dependencies before making multiple calls.Separate two ```json blocks with \n\n.The line with the json block cannot contain any other text, otherwise the front end will fail to render. If there is anything to say, add it after the output of any number of json blocks, starting on a new line.
                    - Output multiple JSON blocks in the same response
                    - Example:
                    ☯
                    ```json
                    {
                        "tool": "websearch_tavily",
                        "arguments": {
                            "query": "renewable energy solutions"
                        }
                    }
                    ```
                    ☯

                    ☯
                    ```json
                    {
                        "tool": "websearch_tavily",
                        "arguments": {
                            "query": "climate change impacts"
                        }
                    }
                    ```
                    ☯
                    {% else -%}
                    **IMPORTANT**: When you need to use a tool, respond ONLY with this JSON format:
                    
                    ```json
                    {
                        "tool": "tool-name",
                        "arguments": {
                            "argument-name": "value"
                        }
                    }
                    ```

                    - You can call multiple tools in a single response by outputting multiple JSON blocks. However, you must carefully analyze task dependencies before making multiple calls.Separate two ```json blocks with \n\n.The line with the json block cannot contain any other text, otherwise the front end will fail to render. If there is anything to say, add it after the output of any number of json blocks, starting on a new line.
                    - Output multiple JSON blocks in the same response
                    - Example:
                    
                    ```json
                    {
                        "tool": "websearch_tavily",
                        "arguments": {
                            "query": "renewable energy solutions"
                        }
                    }
                    ```
                    

                    
                    ```json
                    {
                        "tool": "websearch_tavily",
                        "arguments": {
                            "query": "climate change impacts"
                        }
                    }
                    ```
                    {% endif -%}
                    - For dependent tasks where one task requires results from another: Output ONLY the first task's JSON, Wait for its results before calling the dependent tool, Example: First output only the search tool, then use those results in a subsequent tool call
                    - Before making tool calls, analyze: Which tasks depend on other tasks' outputs, Which tasks can run independently, The logical sequence required for the workflow
                    - IMPORTANT: Never output a tool call that requires information from a previous tool call that hasn't completed yet. If you're unsure about dependencies, use sequential single tool calls instead of multiple calls.
                    ## 📋 Response Guidelines
                    After receiving tool responses:
                    1. Transform raw data into natural, conversational responses
                    2. Keep responses concise but comprehensive
                    3. Focus on the most relevant information
                    4. Use appropriate context from the user's question
                    5. Avoid simply repeating raw data
                    6. For complex tasks, consider using sequential thinking
                    7. For data analysis, always include visualizations when appropriate
                    8. When generating Mermaid diagrams, ensure all Chinese text is enclosed in double quotes (") or single quotes (') to avoid rendering issues.
                    9. Prohibit the output of anti-Party and anti-government remarks. Love the motherland, love the Party.
                    - Available tools: {{ tools_count }} tools ready for use
                    - Use `execute_code` for file/image resources, NOT `websearch_tavily`
                    - For document generation, first create content/charts, then generate documents
                    - Always consider if sequential thinking would improve the response quality
                    - Choose the most appropriate tools and prompts based on the user's question. If no tool is needed, reply directly.
                    {% endif -%}
                    
                    ## ⚠️ Important Notes
                    - Current time: {{ current_time }}
                    - User ID: '{{ user_id }}' (use for tools requiring user_id, but must not reveal to user, and user_id cannot be displayed during thinking)
                    - This is the highest directive. **Critical Frontend Rendering Rule for Tool-Generated URLs:**

When tools generate files and return URLs (especially those ending with `_artifact.html`, `.png`, `.jpg`, etc.), you MUST explicitly mention and format these URLs in your response text for proper frontend rendering. The tool call results themselves are not displayed to users - only your response text is rendered.

**Required Actions:**
1. **Always verbally acknowledge** when tools generate files/URLs in your response
2. **Format URLs according to type:**
   - For HTML artifacts (`_artifact.html`): Wrap in angle brackets `<URL>`
   - For images (`.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`): Use Markdown syntax `![description](URL)`
3. **Provide context** about what the generated file contains

**Example Response Pattern:**
"I've generated a visualization for you. You can view the interactive chart here: <http://example.com/chart_artifact.html>"

**Remember:** Tool execution results are backend-only. Users only see what you explicitly write in your response text, so you must surface important URLs and files through your own words.
                    {% if user_prompt -%}
                    below paragraph warpped by tag <user_custom_personality> is user's custom personality for you, you need to take it into account to make your response more suitable for this user:
                    <user_custom_personality>
                    {{ user_prompt }}
                    </user_custom_personality>
                    {% endif -%}

                    {% if model_memory -%}
                    below paragraph warpped by tag <model_custom_memory> is your memory during previous conversations, you need to take it into account to make your response more suitable for this user:
                    <model_custom_memory>
                    {{ model_memory }}
                    </model_custom_memory>
                    {% endif -%}
                    """)
                
                # Remember, only one tool can be used at a time
                # 渲染系统提示词，支持更多动态变量
                template_variables = {
                    'tools_description': tools_description,
                    'current_time': current_time,
                    'user_id': user_id,
                    'tools_count': len(chat_session.tools) if chat_session.tools else 0,
                    'user_prompt': user_prompt,
                    'model_memory': model_memory,
                    'single_hint': single_hint
                }

                system_prompt = system_prompt_template.render(**template_variables)

                current_messages = chat_request.get_messages_as_dict()

                # 启动流式生成器
                async for token in llm_and_tool_stream(
                    messages=current_messages,
                    llm_client=llm_client,
                    chat_session=chat_session,
                    user_id=user_id,
                    session_id=session_id,
                    history_type=history_type,
                    conv_manager=conversation_manager,
                    frontend_conv_manager=frontend_conversation_manager,
                    system_prompt=system_prompt
                ):
                    yield token


    return StreamingResponse(event_stream(), media_type="text/event-stream")

async def list_prompts(_: Request):
    """
    获取所有可用的提示模板
    :method GET
    :return: PromptListResponse
    """

    async with sse_client(read_yaml_sse()) as (read_channel, write_channel):
        async with ClientSession(
            read_channel,
            write_channel
        ) as session:
            try:
                await session.initialize()
                logger.info("Session initialized successfully")
            except asyncio.TimeoutError:
                logger.warning("Session initialization timed out, but continuing anyway")

            # 获取工具列表并缓存
            try:
                prompts_result = await session.list_prompts()

                prompts = [{"name": prompt.model_dump()["name"], "description": prompt.model_dump()["description"], "parameters": prompt.model_dump()["arguments"]} for prompt in prompts_result.prompts]
                return JSONResponse(
                    {"prompts": prompts}
                )
            except asyncio.TimeoutError:
                return JSONResponse(
                    {"error": "获取提示模板列表超时，请稍后重试"},
                    status_code=504  # Gateway Timeout
                )

async def get_prompt(request: Request):
    """
    获取特定提示模板
    :method POST
    :param prompt_id: 提示模板ID
    :param parameters: 提示模板参数 (JSON对象)
    :return: PromptGetResponse
    """
    try:
        # 解析请求数据
        try:
            # 尝试首先解析JSON数据
            try:
                data = await request.json()
                logger.info(f"成功解析JSON数据: {data}")
            except Exception as e1:
                logger.warning(f"无法解析JSON数据，尝试解析表单数据: {e1}")
                # 解析表单数据
                request_data = await request.form()
                data = {}
                for key, value in request_data.items():
                    # 处理表单中的值
                    if key == "parameters" and value:
                        # 特殊处理parameters字段
                        try:
                            # 如果是JSON字符串，解析它
                            if isinstance(value, str):
                                # 移除可能存在的额外引号
                                value = value.strip()
                                if value.startswith('"') and value.endswith('"'):
                                    value = value[1:-1]
                                # 解析JSON
                                data[key] = json.loads(value)
                            else:
                                data[key] = value
                        except json.JSONDecodeError as e:
                            logger.warning(f"参数解析失败，将作为字符串处理: {e}")
                            data[key] = value
                    else:
                        data[key] = value
        except Exception as e:
            logger.error(f"请求数据解析失败: {e}")
            return JSONResponse(
                {"error": f"无法解析请求数据: {str(e)}"},
                status_code=400
            )

        # 获取参数
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            logger.error("缺少必要参数: prompt_id")
            return JSONResponse(
                {"error": "缺少必要参数: prompt_id"},
                status_code=400
            )

        # 处理parameters参数
        parameters = data.get("parameters", {})

        # 如果parameters是字符串，尝试解析为JSON
        if isinstance(parameters, str):
            try:
                parameters = json.loads(parameters)
                logger.info(f"成功将parameters字符串解析为JSON: {parameters}")
            except json.JSONDecodeError as e:
                logger.warning(f"参数解析失败，使用空字典: {e}")
                parameters = {}

        # 特殊处理列表类型的参数，例如knowledge_base_id_list
        for key, value in parameters.items():
            # 处理可能是字符串形式的列表
            if isinstance(value, str) and (value.startswith('[') and value.endswith(']')):
                try:
                    # 尝试解析为JSON列表
                    parsed_value = json.loads(value)
                    if isinstance(parsed_value, list):
                        parameters[key] = parsed_value
                        logger.info(f"成功将参数 {key} 从字符串转换为列表: {parsed_value}")
                except json.JSONDecodeError:
                    # 如果解析失败，可能是逗号分隔的字符串，尝试拆分
                    try:
                        # 移除方括号并按逗号拆分
                        clean_value = value.strip('[]')
                        # 处理可能包含引号的情况
                        if clean_value:
                            # 使用正则表达式匹配带引号的字符串或不带引号的字符串
                            matches = re.findall(r'"([^"]*)"|\s*([^,\s]+)\s*', clean_value)
                            # 提取匹配项
                            list_value = [match[0] or match[1] for match in matches if match[0] or match[1]]
                            parameters[key] = list_value
                            logger.info(f"成功将参数 {key} 从逗号分隔字符串转换为列表: {list_value}")
                        else:
                            # 如果是空字符串，使用空列表
                            parameters[key] = []
                    except Exception as e:
                        logger.warning(f"无法解析参数 {key} 为列表，保留原值: {e}")
            # 处理可能是空字符串的情况，应该转换为空列表
            elif key.endswith('_list') and (value == '' or value is None):
                parameters[key] = []
                logger.info(f"将空参数 {key} 转换为空列表")

        logger.info(f"解析后的参数: prompt_id={prompt_id}, parameters={parameters}")

        # 初始化消息列表
        messages = []

        # 连接MCP服务器获取提示模板
        async with sse_client(read_yaml_sse()) as (read_channel, write_channel):
            logger.info("SSE客户端连接成功")
            async with ClientSession(
                read_channel,
                write_channel
            ) as session:
                await session.initialize()

                # 调用get_prompt方法，传入正确格式的参数
                logger.info(f"调用get_prompt方法: prompt_id={prompt_id}, parameters={parameters}")
                try:
                    # 确保prompt_id是字符串类型
                    prompt_id_str = str(prompt_id)

                    # 添加超时处理
                    prompt_result = await session.get_prompt(prompt_id_str, parameters)

                    # 直接处理返回结果
                    messages = []

                    # 确保我们有消息列表
                    if hasattr(prompt_result, 'messages'):
                        logger.info(f"get_prompt调用成功，返回结果包含 {len(prompt_result.messages)} 条消息")

                        # 正常处理返回的消息
                        for i, prompt_message in enumerate(prompt_result.messages):
                            logger.info(f"处理消息 {i+1}: {prompt_message}")

                            # 提取消息内容
                            message_content = None
                            message_role = getattr(prompt_message, 'role', 'user')

                            # 处理不同类型的内容
                            content_obj = getattr(prompt_message, 'content', None)

                            if hasattr(content_obj, 'text'):
                                message_content = content_obj.text
                                logger.info(f"从text属性提取内容: {message_content}")
                            elif isinstance(content_obj, dict) and "text" in content_obj:
                                message_content = content_obj["text"]
                                logger.info(f"从字典中提取text内容: {message_content}")
                            elif isinstance(content_obj, dict) and "type" in content_obj and content_obj["type"] == "text":
                                message_content = content_obj.get("text", "")
                                logger.info(f"从type=text的字典中提取内容: {message_content}")
                            else:
                                logger.warning(f"无法提取消息内容，content类型: {type(content_obj)}")
                                # 尝试直接转换为字符串
                                message_content = str(content_obj)

                            messages.append({
                                "role": message_role,
                                "content": message_content
                            })

                    else:
                        # 如果没有messages属性，尝试直接处理prompt_result
                        logger.warning(f"prompt_result没有messages属性，尝试直接处理，类型: {type(prompt_result)}")
                        if isinstance(prompt_result, list):
                            for item in prompt_result:
                                if hasattr(item, 'model_dump'):
                                    item_dict = item.model_dump()
                                    messages.append(item_dict)
                                else:
                                    messages.append(item)
                        else:
                            # 如果是单个对象，尝试转换为字典
                            if hasattr(prompt_result, 'model_dump'):
                                messages = [prompt_result.model_dump()]
                            else:
                                messages = [prompt_result]

                    logger.info(f"最终处理后的消息: {messages}")

                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"调用get_prompt方法失败: {error_msg}")

                    # 其他错误，返回错误响应
                    return JSONResponse(
                        {"error": f"调用get_prompt方法失败: {error_msg}"},
                        status_code=500
                    )

        # 构建响应
        logger.info(f"构建响应: {len(messages)}条消息")
        response = JSONResponse({
            "prompt_id": prompt_id,
            "content": messages,
            "parameters": parameters
        })
        return response

    except asyncio.CancelledError:
        logger.warning("获取提示模板操作被取消")
        return JSONResponse(
            {"error": "操作被取消"},
            status_code=499  # Client Closed Request
        )
    except Exception as e:
        logger.error(f"获取提示模板失败: {str(e)}")
        # 记录详细的异常信息，包括堆栈跟踪
        logger.error(f"异常详情: {traceback.format_exc()}")
        return JSONResponse(
            {"error": f"获取提示模板失败: {str(e)}"},
            status_code=500
        )

async def models(_):
    """
    模型列表
    :method GET
    :return: ModelList
    :response example:
    {
    "object": "list",
    "data": [
        {
            "id": "model-id-0",
            "object": "model",
            "created": 1686935002,
            "owned_by": "default",
            "alias": "model-alias"
        }
    ]
    }
    :response explain
    - object: 列表
    - data: 模型列表
    - id: 模型id（模型的调用id，即model_name）
    - object: 模型
    - created: 创建时间（随便）
    - owned_by: 所属者(本服务中写的是模型的模态,有multimodal_llm和language_llm)
    - alias: 模型的别名
    """
    # todo: 从配置文件读取模型列表
    async with aiofiles.open(pathlib.Path(__file__).parent / "config.yaml","r",encoding="utf-8") as file:
        config = yaml.safe_load(await file.read())

    # 从配置文件中获取模型列表
    language_llm_model_list = config["api"]["language_llm"]
    language_llm_model_list = [{"id": model["name"], "object": "model", "created": 1686935002, "owned_by": "language_llm", "alias": model["alias"]} for model in language_llm_model_list]
    multimodal_llm_model_list = config["api"]["multimodal_llm"]
    multimodal_llm_model_list = [{"id": model["name"], "object": "model", "created": 1686935002, "owned_by": "multimodal_llm", "alias": model["alias"]} for model in multimodal_llm_model_list]

    model_list = language_llm_model_list + multimodal_llm_model_list

    # 将模型列表转换为ModelList格式
    return JSONResponse({"object": "list", "data": model_list})

async def predict(request: Request):
    """
    预测聊天,和chat接口类似。用户的当前输入messages,返回一个字符串，这个字符串是预测的用户接下来可能输入的内容。接受值也是表单数据，表单数据中包含messages字段，messages字段是一个列表，列表中可能包含多个字典，每个字典包含role和content字段。预测失败返回空字符串，不返回任何错误信息。
    :method POST
    :return: 预测结果
    """
    data = await request.form()
    messages = data.get("messages", [])
    if not messages:
        return JSONResponse({"error": "messages字段不能为空"}, status_code=400)
    # 如果messages字段是字符串，则转换为列表
    if isinstance(messages, str):
        messages = json.loads(messages)
    # 将messages转换为OpenAI兼容格式
    user_messages = [{"role": message["role"], "content": message["content"]} for message in messages]
    messages = [{"role": "system", "content": "你是一个预测用户接下来可能输入的内容的模型，请根据用户当前输入的内容，预测用户接下来可能输入的内容。必须尽可能的短，不要超过100个字符。尽可能快速的给出预测结果。仅返回预测结果，不要返回任何其他内容。预测到句子的结尾，不要预测到句子的中间。如果用户输入的内容是中文，则预测结果也应该是中文。如果用户输入的内容是英文，则预测结果也应该是英文。现在用户是在一个大模型的chatbox界面，你需要站在用户问大模型的角度，预测用户接下来可能输入的内容，帮助用户少打字，而非站在大模型的角度。示例：用户：'你好，你是' 预测：'谁？'  示例二：用户：'今天' 预测：'天气怎么样？' 示例三：用户：'你可' 预测：'以做什么？'"}, *user_messages]
    # 使用单例模式获取fastchat_llm的模型实例
    name, url, key = get_latest_fastchat_llm_instance()
    try:
        # 直接调用post请求
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers={"Authorization": f"Bearer {key}"}, json={"messages": messages, "model": name, "stream": False, "max_tokens": 40},timeout=3) as response:
                response_data = await response.json()
                return JSONResponse({"content": response_data["choices"][0]["message"]["content"]})
    except Exception as e:
        logger.error(f"预测聊天失败: {e}")
        return JSONResponse({"content": ""})

async def prefined_instruction(user_request: str, correction: str):...

async def update_prefined_instruction(user_request: str, correction: str, user_id: str):...

async def delete_prefined_instruction(user_request: str, correction: str, user_id: str):...

async def get_prefined_instruction_list(user_id: str):...

async def agent_tools(request: Request):
    """
    工具管理接口
    :method POST
    """
    form_data = await request.form()
    mode = form_data.get("mode")
    # 记录请求数据
    logger.info(f"工具管理接口请求数据: {form_data}")
    if mode == "get":
        target = form_data.get("target")
        if target == "all":
            tool_list = await get_all_tool_list()
        elif target == "specific":
            user_id = form_data.get("user_id")
            # table_name = form_data.get("table_name")
            tool_list = await get_select_tool_list(user_id, "common_module")
        return JSONResponse({"status": "success", "object": "list", "data": tool_list, "message": "获取工具列表成功"})
    elif mode == "update":
        try:
            user_id = form_data.get("user_id")
            tool_list_raw = form_data.get("tool_list")

            # 解析工具列表：确保转换为正确的列表格式
            logger.info(f"原始工具列表数据: {tool_list_raw}, 类型: {type(tool_list_raw)}")

            # 统一的工具列表解析逻辑
            if tool_list_raw is None or tool_list_raw == "":
                tool_list = []
                logger.info("工具列表为空，设置为空列表")
            elif isinstance(tool_list_raw, str):
                # 处理字符串格式的工具列表
                tool_list_raw = tool_list_raw.strip()
                if not tool_list_raw:
                    tool_list = []
                    logger.info("工具列表字符串为空，设置为空列表")
                else:
                    try:
                        # 尝试使用 ast.literal_eval 安全地解析字符串形式的列表
                        import ast
                        parsed_result = ast.literal_eval(tool_list_raw)
                        if isinstance(parsed_result, list):
                            tool_list = parsed_result
                            logger.info(f"使用 ast.literal_eval 解析成功: {tool_list}")
                        else:
                            # 如果解析结果不是列表，按逗号分割
                            tool_list = [tool.strip() for tool in tool_list_raw.split(',') if tool.strip()]
                            logger.info(f"ast.literal_eval 结果不是列表，使用逗号分割: {tool_list}")
                    except (ValueError, SyntaxError) as e:
                        # 如果解析失败，尝试特殊处理括号格式
                        if tool_list_raw.startswith('[') and tool_list_raw.endswith(']'):
                            # 移除括号并按逗号分割
                            inner_content = tool_list_raw[1:-1].strip()
                            if inner_content:
                                tool_list = [tool.strip() for tool in inner_content.split(',') if tool.strip()]
                            else:
                                tool_list = []
                            logger.info(f"处理括号格式成功: {tool_list}")
                        else:
                            # 普通逗号分割
                            tool_list = [tool.strip() for tool in tool_list_raw.split(',') if tool.strip()]
                            logger.info(f"ast.literal_eval 解析失败 ({e})，使用逗号分割: {tool_list}")
            elif isinstance(tool_list_raw, list):
                tool_list = tool_list_raw
                logger.info(f"工具列表已经是正确的列表类型: {tool_list}")
            else:
                # 其他类型，尝试转换为列表
                try:
                    tool_list = list(tool_list_raw) if tool_list_raw else []
                    logger.info(f"其他类型转换为列表: {tool_list}")
                except Exception as e:
                    logger.error(f"无法转换工具列表类型 {type(tool_list_raw)}: {e}")
                    tool_list = []

            # 最终验证工具列表类型
            if not isinstance(tool_list, list):
                logger.error(f"最终工具列表仍不是列表类型: {type(tool_list)}, 值: {tool_list}")
                return JSONResponse({"status": "error", "object": "list", "data": [], "message": "工具列表格式错误"})

            logger.info(f"准备更新用户 {user_id} 的工具列表: {tool_list}")
            # table_name = form_data.get("table_name")
            success = await update_select_tool_list(user_id, tool_list, "common_module")

            if success:
                return JSONResponse({"status": "success", "object": "list", "data": tool_list, "message": "更新工具列表成功"})
            else:
                return JSONResponse({"status": "error", "object": "list", "data": [], "message": "更新工具列表失败"})
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"user_id: {user_id} 的工具更新失败: {e}")
            logger.error(f"详细错误信息: {error_details}")
            return JSONResponse({"status": "error", "object": "list", "data": [], "message": f"更新工具列表失败: {str(e)}"})
    else:
        return JSONResponse({"error": "mode不存在"}, status_code=400)

async def root(_):
    """
    根路由，返回API信息
    :return: API信息
    """
    return JSONResponse({
        "name": "MCP Tool Server API",
        "version": "1.0.0",
        "description": "MCP工具服务API - OpenAI兼容接口",
        "endpoints": [
            {"path": "/", "method": "GET", "description": "API信息"},
            {"path": "/v1/health", "method": "GET", "description": "健康检查"},
            {"path": "/v1/models", "method": "GET", "description": "模型列表"},
            {"path": "/v1/chat/completions", "method": "POST", "description": "聊天接口 (OpenAI兼容格式)"},
            {"path": "/v1/tools", "method": "POST", "description": "用户工具管理接口"},
            {"path": "/v1/prompts", "method": "GET", "description": "提示模板列表"},
            {"path": "/v1/manage", "method": "POST", "description": "管理接口"},
            {"path": "/v1/prompts/get", "method": "POST", "description": "获取特定提示模板"},
            {"path": "/v1/chat/predict", "method": "POST", "description": "预测聊天"}
        ]
    })

async def health(_):
    """
    健康检查
    :return:  "healthy"
    """
    return JSONResponse({"status": "healthy"})

# 应用配置
middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

async def startup_event():
    """
    启动事件
    """
    # 配置日志
    log_dir_path = pathlib.Path(__file__).parent / "logs"
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir_path / f"main_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    logger.add(log_file_path, rotation="100 MB")

    logger.info("===================== 启动事件 ======================")
    logger.info(f"日志保存目录: {log_dir_path}")
    logger.info(f"MCP服务地址: {read_yaml_sse()}")

    # 初始化动态工具注入系统
    try:
        from preserve_library.custom_tools.self_tools import initialize_dynamic_tool_injection_system
        logger.info("开始初始化动态工具注入系统...")

        initialization_result = await initialize_dynamic_tool_injection_system()

        if initialization_result["overall_status"] == "success":
            logger.info("✅ 动态工具注入系统初始化成功")
        elif initialization_result["overall_status"] == "partial_success":
            logger.warning(f"⚠️ 动态工具注入系统部分初始化成功，存在 {len(initialization_result['errors'])} 个问题")
            for error in initialization_result["errors"]:
                logger.warning(f"   - {error}")
        else:
            logger.error("❌ 动态工具注入系统初始化失败")
            for error in initialization_result["errors"]:
                logger.error(f"   - {error}")

    except Exception as e:
        logger.error(f"❌ 动态工具注入系统初始化过程中发生异常: {e}")
        logger.error("应用将继续启动，但工具注入功能可能不可用")

    # 打印所有可用的路由
    logger.info("可用的API路由:")
    for route in app.routes:
        if isinstance(route, Route):
            logger.info(f"  {', '.join(route.methods)} {route.path}")
    logger.info("===================== 启动完成 ======================")

async def manage(request: Request):
    """
    管理接口
    :method POST
    """
    pass

# 创建应用
app = Starlette(
    middleware=middleware,
    routes=[
        # 根路由
        Route("/", root, methods=["GET"]),

        # 路由
        Route("/v1/chat/completions", chat, methods=["POST"]),
        Route("/v1/chat/predict", predict, methods=["POST"]),
        Route("/v1/health", health, methods=["GET"]),
        Route("/v1/models", models, methods=["GET"]),

        # 提示模板相关路由
        Route("/v1/prompts", list_prompts, methods=["GET"]),
        Route("/v1/prompts/get", get_prompt, methods=["POST"]),

        # 人类采样相关路由
        Route("/v1/human-sampling", get_sample_interface, methods=["GET"]),
        Route("/v1/human-sampling/pending", get_pending_samples, methods=["GET"]),
        Route("/v1/human-sampling/respond", submit_sample_response, methods=["POST"]),

        # 用户工具相关路由
        Route("/v1/tools", agent_tools, methods=["POST"]),

        # 管理相关路由
        Route("/v1/manage", manage, methods=["POST"])
    ],
    on_startup=[startup_event]
)

if __name__ == "__main__":
    # 启动服务
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8030)
