import argparse
import os
import re
import json
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
    query_text: Annotated[str, Field(description="Query text to rerank against")],
    search_results: Annotated[
        str, Field(description="This parameter is automatically generated, no need to pass manually")],
    top_n: Annotated[
        int,
        Field(description="Number of top results to return", ge=10, le=30)] = 20,
    threshold: Annotated[
        float,
        Field(description="Minimum relevance score threshold for filtering documents", ge=0.0, le=1.0)] = 0.2,
    ctx: Context = None,
) -> str:
    """对检索结果进行重排序。适用于检索结果较多、需要高 precision 的情况。"""

    # 解析 <source> 格式
    source_pattern = r'<source id="(\d+)">\s*\n(.*?)\n</source>'
    source_matches = re.findall(source_pattern, search_results, re.DOTALL)
    
    if not source_matches:
        return "Error: No valid <source> blocks found in search results"
    
    # 解析每个 source 块
    documents = []
    source_info = []
    doc_to_source = []  # 记录每个文档属于哪个source和在该source中的索引
    
    for source_id, source_content in source_matches:
        try:
            # 解析JSON内容
            source_data = json.loads(source_content.strip())
            source_name = source_data.get("source", {}).get("name", "")
            documents_list = source_data.get("document", [])
            distances_list = source_data.get("distances", [])
            
            source_info.append({
                "id": source_id,
                "name": source_name,
                "documents": documents_list,
                "distances": distances_list
            })
            
            # 将该source的所有文档添加到总文档列表
            for doc_idx, doc in enumerate(documents_list):
                documents.append(doc.strip())
                doc_to_source.append((len(source_info) - 1, doc_idx))  # (source_index, doc_index)
                
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse source {source_id}: {e}")
            continue
    
    if not documents:
        return "Error: No valid documents found in search results"
    
    # 调用 reranker API
    url = f"{reranker_base_url}/rerank"
    headers = {"Content-Type": "application/json"}
    data = {"model": reranker_model, "query": query_text, "documents": documents}
    
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        rerank_results = response.json()
    except requests.exceptions.RequestException as e:
        return f"Error calling reranker API: {str(e)}"
    
    if 'results' not in rerank_results:
        return "Error: Invalid reranker response format"
    
    # 按照 relevance_score 排序
    sorted_results = sorted(rerank_results['results'],
                            key=lambda x: x['relevance_score'],
                            reverse=True)
    
    # 重构source结构，用重排序分数替换distance
    # 首先收集每个source中被选中的文档
    def collect_selected_docs(current_threshold):
        source_selected_docs = {}  # source_index -> [(doc_index, relevance_score)]
        selected_count = 0
        
        for rerank_result in sorted_results:
            if selected_count >= top_n:
                break
                
            doc_global_index = rerank_result['index']
            relevance_score = rerank_result['relevance_score']
            
            # 过滤掉分数低于阈值的文档
            if relevance_score < current_threshold:
                continue
            
            if doc_global_index < len(doc_to_source):
                source_idx, doc_idx = doc_to_source[doc_global_index]
                
                if source_idx not in source_selected_docs:
                    source_selected_docs[source_idx] = []
                source_selected_docs[source_idx].append((doc_idx, relevance_score))
                selected_count += 1
        
        return source_selected_docs, selected_count
    
    # 首次尝试使用原始threshold
    current_threshold = threshold
    source_selected_docs, selected_count = collect_selected_docs(current_threshold)
    
    # 检查source数量是否小于5，如果是则降低threshold重试一次
    if len(source_selected_docs) < 5:
        logger.info(f"Found only {len(source_selected_docs)} sources with threshold {current_threshold:.3f}, adjusting threshold to {current_threshold * 0.6:.3f} for retry")
        current_threshold = current_threshold * 0.6
        source_selected_docs, selected_count = collect_selected_docs(current_threshold)
        logger.info(f"After threshold adjustment, found {len(source_selected_docs)} sources with {selected_count} total documents")
    
    # 重构输出 - 按最大距离分数排序
    output_sources = []
    source_with_max_scores = []
    
    for source_idx in source_selected_docs.keys():
        source = source_info[source_idx]
        selected_docs_info = source_selected_docs[source_idx]
        
        # 按原始顺序排序选中的文档
        selected_docs_info.sort(key=lambda x: x[0])
        
        new_documents = []
        new_distances = []
        
        for doc_idx, relevance_score in selected_docs_info:
            new_documents.append(source["documents"][doc_idx])
            new_distances.append(relevance_score)
        
        # 计算该source的最大距离分数
        max_distance = max(new_distances) if new_distances else 0
        
        # 构建新的source块
        new_source = {
            "source": {
                "name": source["name"]
            },
            "document": new_documents,
            "distances": new_distances
        }
        
        source_with_max_scores.append((max_distance, new_source))
    
    # 按最大距离分数从大到小排序
    source_with_max_scores.sort(key=lambda x: x[0], reverse=True)
    
    # 重新编号并生成输出
    for new_source_id, (max_score, source_data) in enumerate(source_with_max_scores, 1):
        source_json = json.dumps(source_data, ensure_ascii=False, indent=4)
        output_sources.append(f'<source id="{new_source_id}">\n{source_json}\n</source>')
    
    return "\n\n".join(output_sources)


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
