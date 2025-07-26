import asyncio
import logging

import uvicorn
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession,ClientResponse

from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from sub_mcp_server.photo_mcp_server import photo_mcp
from sub_mcp_server.translate_mcp_server import translate_mcp
from sub_mcp_server.web_search_mcp_server import websearch_mcp
from sub_mcp_server.human_sample_server import hitl_app
from sub_mcp_server.knowledge_mcp_server import knowledge_mcp
from sub_mcp_server.sendbox_mcp_server import sandbox_mcp
from mcp.server.fastmcp.prompts import base
from sub_mcp_server.mail_mcp_server import mail_mcp
from sub_mcp_server.sequentialthinking_mcp_server import sequentialthinking_mcp
from sub_mcp_server.model_custom_memory_mcp_server import memory_mcp
from sub_mcp_server.artifact_mcp_server import artifact_mcp
from mcp.types import TextContent
from sub_mcp_server.text2image_mcp_server import text2image_mcp
from sub_mcp_server.await_mcp_server import await_mcp
from sub_mcp_server.text2video_mcp_server import text2video_mcp
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


from typing import List, Any
from mcp.types import Tool as MCPTool
from mcp.types import Prompt as MCPPrompt

main_mcp = FastMCP("主Mcp服务")

# 创建主 FastMCP 的 SSE 应用，挂载根路径
app = main_mcp.sse_app("/")

# 添加一个健康检查端点，用于调试
async def health_check(request):
    results = {}
    for sub in SUB_MCPS:
        try:
            await sub.list_tools(timeout=1.0)  # 测试响应
            results[sub.name] = "active"
        except asyncio.TimeoutError:
            results[sub.name] = "timeout"
    return JSONResponse({"sub_services": results})

# 添加路由
app.routes.append(Route("/health", health_check))
# 统一挂载子服务
SUB_MCPS = [photo_mcp, translate_mcp, websearch_mcp, knowledge_mcp, sandbox_mcp, mail_mcp, sequentialthinking_mcp, memory_mcp, artifact_mcp, text2image_mcp, text2video_mcp, await_mcp]
# 将各子 mcp 以独立的子路径挂载到主应用
# 这样对应的 SSE 路径分别为：
#   /photo/sse           （GET 建立 SSE 连接）
#   /photo/messages/     （POST 发送消息）
# 统一挂载子服务
for sub in SUB_MCPS:
    app.mount(f"/{sub.name}", sub.sse_app(""))
# ------------------------  聚合层：合并子 mcp 的工具  ------------------------
@main_mcp._mcp_server.list_prompts()
async def aggregated_list_prompts() -> List[MCPPrompt]:
    """聚合子 mcp 的 prompt 列表并返回"""
    prompts: List[MCPPrompt] = []
    for sub_mcp in SUB_MCPS:
        sub_prompts = await sub_mcp.list_prompts()
        prompts.extend(sub_prompts)
    return prompts
# ------------------------  聚合层：合并子 mcp 的 prompt ------------------------
@main_mcp._mcp_server.list_tools()
async def aggregated_list_tools() -> List[MCPTool]:
    """聚合子 mcp 的工具列表并返回"""
    tools: List[MCPTool] = []
    # 收集各子 mcp 的工具
    for sub_mcp in SUB_MCPS:
        sub_tools = await sub_mcp.list_tools()
        tools.extend(sub_tools)
    return tools

# ------------------------  聚合层：代理子 mcp 的工具调用 ------------------------
@main_mcp._mcp_server.call_tool()
async def aggregated_call_tool(name: str, arguments: dict):
    """统一代理子 mcp 的工具调用，不做特殊化处理"""
    if sub_mcp := TOOL_MAPPING.get(name):
        return await sub_mcp.call_tool(name, arguments)

    available_tools = ", ".join(TOOL_MAPPING.keys())
    raise ValueError(f"工具 '{name}' 不存在。可用工具: [{available_tools}]")

@main_mcp._mcp_server.get_prompt()
async def aggregated_get_prompt(name: str, arguments: dict[str, Any]):
    """根据提示名称，将调用代理到对应的子 mcp"""
    for sub_mcp in SUB_MCPS:
        sub_prompts = await sub_mcp.list_prompts()
        if any(prompt.name == name for prompt in sub_prompts):
            return await sub_mcp.get_prompt(name, arguments)

    raise ValueError(f"未知提示: {name}")

@app.on_event("startup")
async def startup_event():
    global TOOL_MAPPING
    TOOL_MAPPING = {}  # 格式: {tool_name: sub_mcp_object}
    
    for sub in SUB_MCPS:
        tools = await sub.list_tools()
        for tool in tools:
            # 记录工具名对应的子服务对象
            TOOL_MAPPING[tool.name] = sub
            logger.info(f"注册工具: {tool.name} -> {sub.name}")

if __name__ == "__main__":
    # 启动 Uvicorn 服务，监听 8000 端口
    logger.info("启动MCP服务器，监听 0.0.0.0:8002")
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info",log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "()": "uvicorn.logging.DefaultFormatter",
                    "fmt": "%(asctime)s - PID:%(process)d - %(name)s - %(levelprefix)s %(message)s",
                    "use_colors": True,
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "loggers": {
                "": {"handlers": ["default"], "level": "INFO"},
                "uvicorn.error": {"level": "INFO"},
                "uvicorn.access": {"handlers": ["default"], "level": "INFO", "propagate": False},
            },
        })