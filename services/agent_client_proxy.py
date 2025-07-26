from typing import Any, AsyncIterable, Dict, Union, AsyncGenerator

from a2a_servers.common.client.client import A2AClient
from a2a_servers.common.types import (
    SendTaskStreamingResponse,
    TaskSendParams,
    SendTaskResponse,
)


class AgentClientProxy:
    """在 A2AClient 之上封装的代理层。

    功能：
    1. MCP Tools 注入：在调用时把工具清单写入 `params.metadata['mcp_tools']`。
    2. 流式 / 非流式统一：提供 `call_task(stream)` 接口，无论流式与否返回相同签名，便于上层编程。
    3. 保持 A2AClient 原样未改，降低耦合。"""

    def __init__(self, agent_url: str, mcp_tools: Dict[str, Any] | None = None):
        self.client = A2AClient(url=agent_url)
        self.mcp_tools = mcp_tools or {}

    # ---------------------------------------------------------------------
    # Public unified interface
    # ---------------------------------------------------------------------
    async def call_task(
        self, params: TaskSendParams, *, stream: bool = False
    ) -> Union[SendTaskResponse, AsyncGenerator[SendTaskStreamingResponse, None]]:
        """根据 `stream` 标记自动选择流式或非流式调用。

        - 当 `stream=True` 时返回一个异步生成器（可 `async for`）。
        - 否则直接返回 `SendTaskResponse` 对象。"""

        self._inject_mcp_tools(params)

        if stream:
            # 返回异步生成器以便上层 `async for`
            return self.client.send_task_streaming(payload=params.model_dump())
        else:
            return await self.client.send_task(payload=params.model_dump())

    # ------------------------------------------------------------------
    # Back-compat helpers (可选保留)
    # ------------------------------------------------------------------
    async def send_task_streaming(
        self, params: TaskSendParams
    ) -> AsyncIterable[SendTaskStreamingResponse]:
        """向后兼容的流式接口。"""
        self._inject_mcp_tools(params)
        return self.client.send_task_streaming(payload=params.model_dump())

    async def send_task(self, params: TaskSendParams) -> SendTaskResponse:
        """向后兼容的非流式接口。"""
        self._inject_mcp_tools(params)
        return await self.client.send_task(payload=params.model_dump())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _inject_mcp_tools(self, params: TaskSendParams):
        """若存在 MCP 工具，将其写入 metadata，供下游模型解析。"""
        if not self.mcp_tools:
            return
        if params.metadata is None:
            params.metadata = {}
        # 避免覆盖上游设置
        params.metadata.setdefault("mcp_tools", self.mcp_tools) 