import argparse
import os
import ast
import re
from typing import Annotated
from pydantic import Field
from dotenv import load_dotenv
import requests
from fastmcp import FastMCP, Context
from fastmcp.server.auth import BearerAuthProvider
from fastmcp.server.auth.providers.bearer import RSAKeyPair
from starlette.requests import Request
from starlette.responses import PlainTextResponse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

enable_auth = os.environ.get("ENABLE_AUTH", "false") == "true"
if enable_auth:
    # 动态生成RSA密钥对
    key_pair = RSAKeyPair.generate()
    auth = BearerAuthProvider(public_key=key_pair.public_key)
    token = key_pair.create_token(expires_in_seconds=86400 * 30)
    mcp = FastMCP(name="Reranker",
                  auth=auth,
                  sse_path="/mcp/reranker-sse/sse",
                  message_path="/mcp/reranker-sse/message/")
    logger.info("Authentication enabled")
else:
    mcp = FastMCP(name="Reranker",
                  sse_path="/mcp/reranker-sse/sse",
                  message_path="/mcp/reranker-sse/message/")
    logger.info("Authentication disabled")


@mcp.tool()
async def rerank(
    search_results: Annotated[
        str, Field(description="Search results from search tools")],
    top_n: Annotated[
        int,
        Field(description="Number of top results to return", ge=1, le=20)] = 5,
    ctx: Context = None,
) -> str:
    """根据查询文本对文档列表进行重排序。适用于检索结果较多、较复杂的情况。
    
    注意：

    1. 必须将 law-searcher 或 case-searcher 的 search 系列工具返回的字符串原封不动地传入到
        search_results 参数，例如 "Sparse vector search results for '行纪合同':\n\n{'chunk_id': ...
    """

    # 解析搜索结果
    parts = search_results.split("\n\n")
    if len(parts) < 2:
        return "Error: Invalid search results format"

    # 提取 query 从第一行
    first_line = parts[0]
    query_match = re.search(r"'([^']+)'", first_line)
    if not query_match:
        return "Error: Could not extract query from search results"

    query = query_match.group(1)

    # 解析每个结果并提取 chunk
    documents = []
    original_results = []

    for part in parts[1:]:
        if part.strip():
            try:
                # 使用 ast.literal_eval 安全解析字典字符串
                result_dict = ast.literal_eval(part.strip())
                chunk = result_dict.get('entity', {}).get('chunk', '')
                if chunk:
                    documents.append(chunk.strip())
                    original_results.append(result_dict)
            except (ValueError, SyntaxError) as e:
                # 如果解析失败，跳过这个结果
                continue

    if not documents:
        return "Error: No valid documents found in search results"

    # 调用 reranker API
    url = f"{reranker_base_url}/rerank"
    headers = {"Content-Type": "application/json"}

    data = {"model": reranker_model, "query": query, "documents": documents}

    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        rerank_results = response.json()
    except requests.exceptions.RequestException as e:
        return f"Error calling reranker API: {str(e)}"

    # 解析重排序结果并添加 relevance_score
    if 'results' not in rerank_results:
        return "Error: Invalid reranker response format"

    # 按照 relevance_score 排序并取前 top_n 个
    sorted_results = sorted(rerank_results['results'],
                            key=lambda x: x['relevance_score'],
                            reverse=True)[:top_n]

    # 构建输出结果
    output = f"Reranking results for '{query}':\n\n"

    for rerank_result in sorted_results:
        index = rerank_result['index']
        relevance_score = rerank_result['relevance_score']

        if index < len(original_results):
            # 创建结果副本并添加 relevance_score
            result_copy = original_results[index].copy()
            result_copy['relevance_score'] = relevance_score

            output += f"{result_copy}\n\n"

    return output


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    return PlainTextResponse("OK")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Reranker MCP Server")
    parser.add_argument("--sse", action="store_true", help="Enable SSE mode")
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()
    args = parse_arguments()

    reranker_base_url = os.environ.get("RERANKER_BASE_URL")
    reranker_model = os.environ.get("RERANKER_MODEL")

    if args.sse:
        if enable_auth:
            logger.info("Token for Auth: %s", token)
        else:
            logger.info("Authentication disabled - no token required")
        mcp.run(transport="sse", port=8000, host="0.0.0.0")
    else:
        if enable_auth:
            logger.info("Token for Auth: %s", token)
        else:
            logger.info("Authentication disabled - no token required")
        mcp.run(transport="streamable-http", port=8000, host="0.0.0.0")
