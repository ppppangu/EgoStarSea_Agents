import asyncio
import logging
from typing import Dict, Any

from a2a_servers.common.server.server import A2AServer
from a2a_servers.common.types import AgentCard
from a2a_servers.common.agent_task_manager import AgentTaskManager
import litellm

logger = logging.getLogger(__name__)


class SimpleChatAgent:
    """简单封装 Litellm，用 OpenAI / Azure / 兼容模型完成对话与流式。"""

    SUPPORTED_CONTENT_TYPES = ["text"]

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model

    async def invoke(self, query: str, session_id: str | None = None) -> str:
        """一次性调用（非流式）。"""
        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": query}],
        )
        return response.choices[0].message.content

    async def stream(self, query: str, session_id: str | None = None):
        """流式调用，逐 token 产生 updates。"""
        content_accum = ""
        async for chunk in litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": query}],
            stream=True,
        ):
            delta = chunk.choices[0].delta.content or ""
            if delta:
                content_accum += delta
                yield {
                    "updates": delta,
                    "is_task_complete": False,
                    "content": None,
                }
        # 完成
        yield {
            "is_task_complete": True,
            "content": content_accum,
            "updates": "",
        }


class AgentLauncher:
    """根据数据库记录启动 / 停止 Agent Server。

    目前示例实现仅在同一进程中以 asyncio.create_task 的形式启动 A2AServer。
    在生产环境可以替换为 Subprocess、Container 或 Kubernetes Pod。"""

    def __init__(self):
        # user_email -> asyncio.Task
        self._running: Dict[str, asyncio.Task] = {}

    async def ensure_agent_running(self, record: Dict[str, Any]):
        """确保某用户的 Agent Server 已经启动。"""
        email = record["user_email"]
        activate = record["activate"]

        if activate:
            if email in self._running and not self._running[email].done():
                return  # already running
            logger.info(f"Launching agent for {email} …")
            task = asyncio.create_task(self._start_agent(record))
            self._running[email] = task
        else:
            # Should not be here normally; handled by ensure_agent_stopped
            await self.ensure_agent_stopped(email)

    async def ensure_agent_stopped(self, email: str):
        task = self._running.get(email)
        if task and not task.done():
            logger.info(f"Stopping agent for {email} …")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self._running[email]

    async def _start_agent(self, record: Dict[str, Any]):
        """在单独任务中运行一个最小的 A2A Server 实例。"""
        # 生成简单的 AgentCard，可按需扩展
        email = record["user_email"]
        export_urls = record.get("export_agent_url", [])
        port = self._derive_port_from_email(email)
        agent_card = AgentCard(
            name=f"Agent-{email}",
            description="Auto-generated agent server",
            url=f"http://localhost:{port}/",
            capabilities={"streaming": True},
        )

        # 使用 AgentTaskManager + SimpleChatAgent 替换 Echo
        chat_agent = SimpleChatAgent()
        task_manager = AgentTaskManager(chat_agent)
        server = A2AServer(host="0.0.0.0", port=port, agent_card=agent_card, task_manager=task_manager)
        await server.astart()

    @staticmethod
    def _derive_port_from_email(email: str) -> int:
        base = 6000
        return base + (abs(hash(email)) % 1000) 