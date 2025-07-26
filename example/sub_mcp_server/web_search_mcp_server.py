"""联网搜索mcp子服务器"""
# 本模块调试通了，禁止任何修改
from mcp.server.fastmcp import FastMCP
from pydantic import Field
import aiohttp
from loguru import logger
import os
import datetime
import json
import pytz
from singleton_mode.singleton_web_search_tavily import get_latest_tavily_api_instance
from sub_mcp_server import ROOT_DIR
from mcp.server.fastmcp.prompts import base
import chardet


websearch_mcp = FastMCP("提供联网搜索功能的mcp工具，可选功能有使用tavilyAPI进行联网搜索")

# 获取当前日期，格式化为YYYY-MM-DD
today = datetime.datetime.now().strftime("%Y-%m-%d")

# 确保日志目录存在
log_dir = f"{ROOT_DIR}/logs/sub_mcp_server"
os.makedirs(log_dir, exist_ok=True)

logger.add(f"{ROOT_DIR}/logs/sub_mcp_server/web_search_mcp_server_{today}.log", level="INFO", format="{time} {level} {message}")

# Web search prompt template using Jinja2
WEB_SEARCH_PROMPT_TEMPLATE = """[WebSearchOnly]You are a professional web search assistant. For the user's question, you MUST use web search tools to find real-time information. Current time: {{ current_time }}.

## Core Instructions:
1. **Primary Tool**: Use `websearch_tavily` for text-based information retrieval
2. **Fallback Strategy**: If `websearch_tavily` fails or returns errors, use `execute_code` tool for web scraping
3. **Resource Handling**: For URLs, files, images, or complex web content, prioritize `execute_code` over `websearch_tavily`

## When to Use execute_code:
- Tavily API returns errors or empty results
- Need to download files (PDF, images, documents)
- Parse complex web structures or tables
- Access sites requiring special headers
- Extract data from specific URLs provided by user

## Code Templates for execute_code:

### Basic Web Scraping:
```python
import requests
from bs4 import BeautifulSoup

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
response = requests.get(url, headers=headers, timeout=30)
soup = BeautifulSoup(response.content, 'html.parser')
title = soup.find('title').text if soup.find('title') else 'No title'
content = soup.get_text(strip=True)
print(f"Title: {title}")
print(f"Content: {content[:1000]}...")
```

### File Download:
```python
import requests
def download_file(url, filename=None):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    response = requests.get(url, headers=headers, stream=True)
    if not filename:
        filename = url.split('/')[-1] or 'downloaded_file'
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded: {filename}")
    return filename
```

### Table Extraction:
```python
import pandas as pd
import requests
response = requests.get(url, headers=headers)
tables = pd.read_html(response.content)
for i, table in enumerate(tables):
    filename = f"table_{i}.csv"
    table.to_csv(filename, index=False)
    print(f"Saved: {filename}")
```

## Response Guidelines:
1. **Structure**: Direct answer first, then detailed explanation (separated by single horizontal line)
2. **Citations**: Include URLs inline using [source](URL) format
3. **References**: End with "References" section listing key sources
4. **Conflicts**: Address conflicting information with balanced analysis
5. **Recency**: Prioritize recent results for time-sensitive topics
6. **Language**: Respond in Chinese unless user requests otherwise
7. **Completeness**: Provide comprehensive, self-contained answers

## Search Strategy:
- Break complex queries into multiple targeted searches
- Use multiple search terms for comprehensive coverage
- Cross-reference information from different sources
- If web search fails completely, honestly inform the user

Remember: You must rely ONLY on web search results, not built-in knowledge."""

@websearch_mcp.prompt('WebSearchOnly')
async def web_search_only():
    """Web search only prompt template - uses only web search without local knowledge base"""
    # Get current time
    current_time = datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')

    # Use Jinja2 template for dynamic content injection
    try:
        from jinja2 import Template
        template = Template(WEB_SEARCH_PROMPT_TEMPLATE)
        prompt = template.render(current_time=current_time)
    except ImportError:
        # Fallback if Jinja2 is not available
        prompt = WEB_SEARCH_PROMPT_TEMPLATE.replace("{{ current_time }}", current_time)

    return [
        base.UserMessage(prompt)
    ]


@websearch_mcp.tool()
async def websearch_tavily(
    query:str = Field(description="要进行搜索的内容，拆分成多个关键词，关键词之间用空格分隔开效果更好"),
    topk:int = Field(description="对于关键词要搜索的条目数量，输入整数数字即可")
):
    """使用tavilyAPI对关键词进行联网搜索"""
    # 确保查询内容是UTF-8编码
    try:
        # 简化处理，确保query是字符串
        query_utf8 = query if isinstance(query, str) else str(query)

        # 记录原始查询和处理后的查询，用于调试
        logger.info(f"原始查询: {repr(query)}")
        logger.info(f"处理后查询: {repr(query_utf8)}")
        logger.info(f"联网搜索工具调用：搜索内容: {query_utf8}, 条目数量: {topk}")
        logger.info(f"使用tavilyAPI对关键词进行联网搜索 - 关键词: {query_utf8}, 原始关键词长度: {len(str(query))}, 条目数量: {topk}")
    except Exception as e:
        logger.error(f"处理查询字符串时出错: {str(e)}")
        query_utf8 = query  # 回退到原始值

    # 对日志和返回值使用ASCII兼容的版本
    api = get_latest_tavily_api_instance()
    if not api:
        error_response = {"error": "Tavily API key不可用"}
        return json.dumps(error_response, ensure_ascii=False)
    # 确保topk是整数
    topk_int = int(topk)

    headers = {
        "Content-Type":"application/json",
        "Authorization": f"Bearer {api}"
    }

    # 确保发送到API的查询是有效的UTF-8
    data = {
        "query": query_utf8,
        "max_results": topk_int
    }

    async with aiohttp.ClientSession() as client:
        async with client.request("post","https://api.tavily.com/search", headers=headers, json=data) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"API请求失败，状态码: {response.status}, 错误: {error_text}")
                error_response = {"error": f"API请求失败，状态码: {response.status}, 错误: {error_text}"}
                return json.dumps(error_response, ensure_ascii=False)

            try:
                # 获取文本内容并手动指定编码为 UTF-8
                response_text = await response.text(encoding='utf-8')
                response_json = json.loads(response_text)

                # 处理搜索结果中的内容编码
                if 'results' in response_json and isinstance(response_json['results'], list):
                    for result in response_json['results']:
                        if 'content' in result and isinstance(result['content'], str):
                            try:
                                # 检查内容是否包含乱码字符
                                if any(c == '�' for c in result['content']) or '\\u' in result['content'] or '' in result['content']:
                                    # 检测编码
                                    raw_bytes = result['content'].encode('utf-8', errors='ignore')
                                    detection = chardet.detect(raw_bytes)
                                    if detection['encoding'] and detection['encoding'] != 'utf-8':
                                        # 尝试用检测到的编码重新解码
                                        result['content'] = raw_bytes.decode(detection['encoding'], errors='replace')
                                    else:
                                        # 如果无法确定编码，用标题或 URL 替代
                                        result['content'] = f"标题: {result.get('title', '未知')}"
                                        if result.get('url'):
                                            result['content'] += f" (来源: {result['url']})"
                            except Exception as e:
                                logger.error(f"处理搜索结果内容编码时出错: {str(e)}")
                                result['content'] = "[内容解码失败]"

                response_str = json.dumps(response_json, ensure_ascii=False)
                logger.info(f"联网搜索工具调用：搜索内容: {query_utf8}，条目数量: {topk}，搜索结果获取成功，搜索内容：{response_str}")
                return response_str

            except Exception as e:
                logger.error(f"处理搜索结果时出错: {str(e)}")
                import traceback
                logger.error(f"异常详情: {traceback.format_exc()}")
                error_response = {"error": f"处理搜索结果时出错: {str(e)}"}
                return json.dumps(error_response, ensure_ascii=False)


if __name__ == "__main__":
    websearch_mcp.run(transport="sse")