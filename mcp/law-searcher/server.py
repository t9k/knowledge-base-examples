import argparse
import os
from contextlib import asynccontextmanager
import logging
from typing import Any, AsyncIterator, Optional, List
from dotenv import load_dotenv
from fastmcp import FastMCP, Context
from fastmcp.server.auth import BearerAuthProvider
from fastmcp.server.auth.providers.bearer import RSAKeyPair
from starlette.requests import Request
from starlette.responses import PlainTextResponse
import uvicorn
from pymilvus import (
    connections,
    Collection,
    AnnSearchRequest,
    RRFRanker,
)
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
import torch
import torch_gcu

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ef = BGEM3EmbeddingFunction(use_fp16=False, device="gcu")


class MilvusConnector:

    def __init__(self,
                 uri: str,
                 token: Optional[str] = None,
                 db_name: Optional[str] = "default",
                 criminal_law_collection_name: Optional[str] = "criminal_law",
                 civil_code_collection_name: Optional[str] = "civil_code"):
        self.uri = uri
        self.token = token
        connections.connect(uri=uri, token=token, db_name=db_name)
        self.criminal_law_collection = Collection(criminal_law_collection_name)
        self.civil_code_collection = Collection(civil_code_collection_name)
        self.check_collections()
        self.criminal_law_collection.load()
        self.civil_code_collection.load()

        self.output_fields = [
            "chunk", "law", "part", "chapter", "section", "article"
        ]

    def check_collections(self) -> None:
        """Check if collections have the expected schema."""
        expected_fields = [
            {
                'name': 'chunk_id',
                'type': 21
            },  # VARCHAR
            {
                'name': 'chunk',
                'type': 21
            },  # VARCHAR
            {
                'name': 'law',
                'type': 21
            },  # VARCHAR
            {
                'name': 'part',
                'type': 21
            },  # VARCHAR
            {
                'name': 'chapter',
                'type': 21
            },  # VARCHAR
            {
                'name': 'section',
                'type': 21
            },  # VARCHAR
            {
                'name': 'article',
                'type': 5
            },  # INT64
            {
                'name': 'article_amended',
                'type': 5
            },  # INT64
            {
                'name': 'sparse_vector',
                'type': 104
            },  # SPARSE_FLOAT_VECTOR
            {
                'name': 'dense_vector',
                'type': 101
            },  # FLOAT_VECTOR
        ]

        # Check both collections
        for collection in [
                self.criminal_law_collection, self.civil_code_collection
        ]:
            schema = collection.describe()
            actual_fields = schema['fields']

            # Check each field
            for expected_field in expected_fields:
                found = False
                for actual_field in actual_fields:
                    if actual_field['name'] == expected_field['name']:
                        found = True
                        # Check field type
                        if actual_field['type'].value != expected_field[
                                'type']:
                            raise ValueError(
                                f"Field {expected_field['name']} has wrong type in {schema['collection_name']} collection. "
                                f"Expected {expected_field['type']}, got {actual_field['type'].value}"
                            )
                        break
                if not found:
                    raise ValueError(
                        f"Missing field {expected_field['name']} in {schema['collection_name']} collection"
                    )

    async def query(self,
                    collection: Collection,
                    filter_expr: str,
                    limit: int = 10) -> list[dict]:
        """Query collection using filter expressions."""
        try:
            return collection.query(expr=filter_expr,
                                    output_fields=self.output_fields,
                                    limit=limit)
        except Exception as e:
            raise ValueError(f"Query failed: {str(e)}")

    async def sparse_search(self,
                            collection: Collection,
                            query_text: str,
                            limit: int = 5,
                            filter_expr: Optional[str] = None) -> list[dict]:
        """
        Perform sparse vector search on a collection.

        Args:
            collection: Collection to search
            query_text: Text to search for
            limit: Maximum number of results
            filter_expr: Optional filter expression
        """
        try:
            query_embeddings = ef([query_text])
            sparse_embedding = query_embeddings["sparse"][[0]]

            sparse_search_params = {"metric_type": "IP", "params": {}}
            results = collection.search(data=[sparse_embedding],
                                        anns_field="sparse_vector",
                                        param=sparse_search_params,
                                        limit=limit,
                                        filter=filter_expr,
                                        output_fields=self.output_fields)[0]
            return results
        except Exception as e:
            raise ValueError(f"Search failed: {str(e)}")

    async def dense_search(self,
                           collection: Collection,
                           query_text: str,
                           limit: int = 5,
                           filter_expr: Optional[str] = None) -> list[dict]:
        """
        Perform dense vector search on a collection.

        Args:
            collection: Collection to search
            query_text: Text to search for
            limit: Maximum number of results
            filter_expr: Optional filter expression
        """
        try:
            query_embeddings = ef([query_text])
            dense_embedding = query_embeddings["dense"][0]

            dense_search_params = {"metric_type": "COSINE", "params": {}}
            results = collection.search(data=[dense_embedding],
                                        anns_field="dense_vector",
                                        param=dense_search_params,
                                        limit=limit,
                                        filter=filter_expr,
                                        output_fields=self.output_fields)[0]
            return results
        except Exception as e:
            raise ValueError(f"Vector search failed: {str(e)}")

    async def hybrid_search(self,
                            collection: Collection,
                            query_text: str,
                            limit: int = 5,
                            filter_expr: Optional[str] = None) -> list[dict]:
        """
        Perform hybrid search combining sparse vector search and dense vector search with RRF ranking.

        Args:
            collection: Collection to search
            query_text: Text to search for
            limit: Maximum number of results
            filter_expr: Optional filter expression
        """
        try:
            query_embeddings = ef([query_text])
            sparse_embedding = query_embeddings["sparse"][[0]]
            dense_embedding = query_embeddings["dense"][0]

            sparse_search_params = {"metric_type": "IP", "params": {}}
            dense_search_params = {"metric_type": "COSINE", "params": {}}
            sparse_request = AnnSearchRequest(data=[sparse_embedding],
                                              anns_field="sparse_vector",
                                              param=sparse_search_params,
                                              limit=limit,
                                              expr=filter_expr)
            dense_request = AnnSearchRequest(data=[dense_embedding],
                                             anns_field="dense_vector",
                                             param=dense_search_params,
                                             limit=limit,
                                             expr=filter_expr)
            results = collection.hybrid_search(
                [sparse_request, dense_request],
                rerank=RRFRanker(60),
                limit=limit,
                output_fields=self.output_fields)[0]

            for result in results:
                result["distance"] = result["distance"] / (2 / (60 + 1))

            return results

        except Exception as e:
            raise ValueError(f"Hybrid search failed: {str(e)}")


class MilvusContext:

    def __init__(self, connector: MilvusConnector):
        self.connector = connector


@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[MilvusContext]:
    """Manage application lifecycle for Milvus connector."""
    config = server.config

    connector = MilvusConnector(
        uri=config.get("milvus_uri", "http://localhost:19530"),
        token=config.get("milvus_token"),
        db_name=config.get("db_name", "default"),
        criminal_law_collection_name=config.get("criminal_law_collection_name",
                                                "criminal_law"),
        civil_code_collection_name=config.get("civil_code_collection_name",
                                              "civil_code"),
    )

    try:
        yield MilvusContext(connector)
    finally:
        pass


key_pair = RSAKeyPair.generate()
auth = BearerAuthProvider(public_key=key_pair.public_key)
mcp = FastMCP(name="Milvus", lifespan=server_lifespan, auth=auth)
token = key_pair.create_token()


@mcp.tool()
async def criminal_law_query(
    filter_expr: str,
    limit: int = 10,
    ctx: Context = None,
) -> str:
    """
    使用过滤表达式查询刑法 Collection。

    Collection 字段（元数据）说明：

    - law (VARCHAR): 刑法及其修正案的名称
    - part (VARCHAR): 编
    - chapter (VARCHAR): 章
    - section (VARCHAR): 节
    - article (INT64): 条
    - article_amended (INT64): 所修改的刑法中的条（仅适用于刑法修正案）

    注意：
    
    1. 刑法有十二部修正案，分别名为"中华人民共和国刑法修正案"、"中华人民共和国刑法修正案（二）"、
    "中华人民共和国刑法修正案（三）"、……、"中华人民共和国刑法修正案（十二）"。

    Args:
        filter_expr: Filter expression, e.g. 'law == "中华人民共和国刑法" and article == 123',
            'law like "%十一%" and article_amended == 338'
        limit: Maximum number of results to return
    """
    connector = ctx.request_context.lifespan_context.connector
    results = await connector.query(
        collection=connector.criminal_law_collection,
        filter_expr=filter_expr,
        limit=limit)

    output = f"Query results for '{filter_expr}':\n\n"
    for result in results:
        output += f"{result}\n\n"

    return output


@mcp.tool()
async def criminal_law_sparse_search(
    query_text: str,
    limit: int = 5,
    filter_expr: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """
    稀疏检索刑法 Collection。适用于专有名词。

    Collection 字段（元数据）说明：

    - law (VARCHAR): 刑法及其修正案的名称
    - part (VARCHAR): 编
    - chapter (VARCHAR): 章
    - section (VARCHAR): 节
    - article (INT64): 条
    - article_amended (INT64): 所修改的刑法中的条（仅适用于刑法修正案）

    注意：
    
    1. 刑法有十二部修正案，分别名为"中华人民共和国刑法修正案"、"中华人民共和国刑法修正案（二）"、
    "中华人民共和国刑法修正案（三）"、……、"中华人民共和国刑法修正案（十二）"。

    Args:
        query_text: Text to search for
        limit: Maximum number of results to return
        filter_expr: Optional filter expression for metadata filtering, e.g. 'chapter like "第三章%"', 'section == "%走私罪%"'
    """
    connector = ctx.request_context.lifespan_context.connector
    results = await connector.sparse_search(
        collection=connector.criminal_law_collection,
        query_text=query_text,
        limit=limit,
        filter_expr=filter_expr)

    output = f"Sparse vector search results for '{query_text}':\n\n"
    for result in results:
        output += f"{result}\n\n"

    return output


@mcp.tool()
async def criminal_law_dense_search(
    query_text: str,
    limit: int = 5,
    filter_expr: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """
    密集检索刑法 Collection。适用于基于语义的检索。

    Collection 字段（元数据）说明：

    - law (VARCHAR): 刑法及其修正案的名称
    - part (VARCHAR): 编
    - chapter (VARCHAR): 章
    - section (VARCHAR): 节
    - article (INT64): 条
    - article_amended (INT64): 所修改的刑法中的条（仅适用于刑法修正案）

    注意：
    
    1. 刑法有十二部修正案，分别名为"中华人民共和国刑法修正案"、"中华人民共和国刑法修正案（二）"、
    "中华人民共和国刑法修正案（三）"、……、"中华人民共和国刑法修正案（十二）"。

    Args:
        query_text: Text to search for
        limit: Maximum number of results to return
        filter_expr: Optional filter expression for metadata filtering, e.g. 'chapter like "第三章%"', 'section == "%走私罪%"'
    """
    connector = ctx.request_context.lifespan_context.connector
    results = await connector.dense_search(
        collection=connector.criminal_law_collection,
        query_text=query_text,
        limit=limit,
        filter_expr=filter_expr)

    output = f"Dense vector search results for '{query_text}':\n\n"
    for result in results:
        output += f"{result}\n\n"

    return output


@mcp.tool()
async def criminal_law_hybrid_search(
    query_text: str,
    limit: int = 5,
    filter_expr: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """
    混合检索刑法 Collection。适用于大多数情况。

    Collection 字段（元数据）说明：

    - law (VARCHAR): 刑法及其修正案的名称
    - part (VARCHAR): 编
    - chapter (VARCHAR): 章
    - section (VARCHAR): 节
    - article (INT64): 条
    - article_amended (INT64): 所修改的刑法中的条（仅适用于刑法修正案）

    注意：
    
    1. 刑法有十二部修正案，分别名为"中华人民共和国刑法修正案"、"中华人民共和国刑法修正案（二）"、
    "中华人民共和国刑法修正案（三）"、……、"中华人民共和国刑法修正案（十二）"。

    Args:
        query_text: Text to search for
        limit: Maximum number of results to return
        filter_expr: Optional filter expression for metadata filtering, e.g. 'chapter like "第三章%"', 'section == "%走私罪%"'
    """
    connector = ctx.request_context.lifespan_context.connector

    results = await connector.hybrid_search(
        collection=connector.criminal_law_collection,
        query_text=query_text,
        limit=limit,
        filter_expr=filter_expr,
    )

    output = f"Hybrid search results for text '{query_text}':\n\n"
    for result in results:
        output += f"{result}\n\n"

    return output


@mcp.tool()
async def civil_code_query(
    filter_expr: str,
    limit: int = 10,
    ctx: Context = None,
) -> str:
    """
    使用过滤表达式查询民法典 Collection。

    Collection 字段（元数据）说明：

    - law (VARCHAR): 民法典的名称
    - part (VARCHAR): 编（包含分编）
    - chapter (VARCHAR): 章
    - section (VARCHAR): 节
    - article (INT64): 条

    注意：
    
    1. 民法典分为"第一编 总则"、"第二编 物权编"、"第三编 合同编"、"第四编 人格权编"、"第五编 婚姻家庭编"、
    "第六编 继承编"、"第七编 侵权责任编"、"第八编 附则"，其中第二编、第三编下还有分编，例如
    "第二编 物权编 第一分编 通则"。

    Args:
        filter_expr: Filter expression, e.g. 'part == "第一编 总则" and article == 123',
            'part like "%第二编%第二分编%" and article == 345'
        limit: Maximum number of results to return
    """
    connector = ctx.request_context.lifespan_context.connector
    results = await connector.query(collection=connector.civil_code_collection,
                                    filter_expr=filter_expr,
                                    limit=limit)

    output = f"Query results for '{filter_expr}':\n\n"
    for result in results:
        output += f"{result}\n\n"

    return output


@mcp.tool()
async def civil_code_sparse_search(
    query_text: str,
    limit: int = 10,
    filter_expr: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """
    稀疏检索民法典 Collection。适用于专有名词。

    Collection 字段（元数据）说明：

    - law (VARCHAR): 民法典的名称
    - part (VARCHAR): 编（包含分编）
    - chapter (VARCHAR): 章
    - section (VARCHAR): 节
    - article (INT64): 条

    注意：
    
    1. 民法典分为"第一编 总则"、"第二编 物权编"、"第三编 合同编"、"第四编 人格权编"、"第五编 婚姻家庭编"、
    "第六编 继承编"、"第七编 侵权责任编"、"第八编 附则"，其中第二编、第三编下还有分编，例如
    "第二编 物权编 第一分编 通则"。

    2. limit 参数默认取 10，当需要针对某个问题或主题，检索全面、详细的信息时，可以设为 20。

    Args:
        query_text: Text to search for
        limit: Maximum number of results to return.
        filter_expr: Optional filter expression for metadata filtering, e.g. 'part like "%婚姻家庭编%"', 'chapter like "%肖像权%"'
    """
    connector = ctx.request_context.lifespan_context.connector
    results = await connector.sparse_search(
        collection=connector.civil_code_collection,
        query_text=query_text,
        limit=limit,
        filter_expr=filter_expr)

    output = f"Sparse vector search results for '{query_text}':\n\n"
    for result in results:
        output += f"{result}\n\n"

    return output


@mcp.tool()
async def civil_code_dense_search(
    query_text: str,
    limit: int = 10,
    filter_expr: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """
    密集检索民法典 Collection。适用于基于语义的检索。

    Collection 字段（元数据）说明：

    - law (VARCHAR): 民法典的名称
    - part (VARCHAR): 编（包含分编）
    - chapter (VARCHAR): 章
    - section (VARCHAR): 节
    - article (INT64): 条

    注意：
    
    1. 民法典分为"第一编 总则"、"第二编 物权编"、"第三编 合同编"、"第四编 人格权编"、"第五编 婚姻家庭编"、
    "第六编 继承编"、"第七编 侵权责任编"、"第八编 附则"，其中第二编、第三编下还有分编，例如
    "第二编 物权编 第一分编 通则"。

    2. limit 参数默认取 10，当需要针对某个问题或主题，检索全面、详细的信息时，可以设为 20。

    Args:
        query_text: Text to search for
        limit: Maximum number of results to return.
        filter_expr: Optional filter expression for metadata filtering, e.g. 'part like "%婚姻家庭编%"', 'chapter like "%肖像权%"'
    """
    connector = ctx.request_context.lifespan_context.connector
    results = await connector.dense_search(
        collection=connector.civil_code_collection,
        query_text=query_text,
        limit=limit,
        filter_expr=filter_expr)

    output = f"Dense vector search results for '{query_text}':\n\n"
    for result in results:
        output += f"{result}\n\n"

    return output


@mcp.tool()
async def civil_code_hybrid_search(
    query_text: str,
    limit: int = 10,
    filter_expr: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """
    混合检索民法典 Collection。适用于大多数情况。

    Collection 字段（元数据）说明：

    - law (VARCHAR): 民法典的名称
    - part (VARCHAR): 编（包含分编）
    - chapter (VARCHAR): 章
    - section (VARCHAR): 节
    - article (INT64): 条

    注意：
    
    1. 民法典分为"第一编 总则"、"第二编 物权编"、"第三编 合同编"、"第四编 人格权编"、"第五编 婚姻家庭编"、
    "第六编 继承编"、"第七编 侵权责任编"、"第八编 附则"，其中第二编、第三编下还有分编，例如
    "第二编 物权编 第一分编 通则"。

    2. limit 参数默认取 10，当需要针对某个问题或主题，检索全面、详细的信息时，可以设为 20。

    Args:
        query_text: Text to search for
        limit: Maximum number of results to return.
        filter_expr: Optional filter expression for metadata filtering, e.g. 'part like "%婚姻家庭编%"', 'chapter like "%肖像权%"'
    """
    connector = ctx.request_context.lifespan_context.connector

    results = await connector.hybrid_search(
        collection=connector.civil_code_collection,
        query_text=query_text,
        limit=limit,
        filter_expr=filter_expr,
    )

    output = f"Hybrid search results for text '{query_text}':\n\n"
    for result in results:
        output += f"{result}\n\n"

    return output


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    return PlainTextResponse("OK")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Milvus MCP Server")
    parser.add_argument("--sse", action="store_true", help="Enable SSE mode")
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_arguments()
    mcp.config = {
        "milvus_uri":
        os.environ.get("MILVUS_URI"),
        "milvus_token":
        os.environ.get("MILVUS_TOKEN"),
        "db_name":
        os.environ.get("MILVUS_DB"),
        "criminal_law_collection_name":
        os.environ.get("MILVUS_COLLECTION_CRIMINAL_LAW"),
        "civil_code_collection_name":
        os.environ.get("MILVUS_COLLECTION_CIVIL_CODE"),
    }

    if args.sse:
        sse_app = mcp.http_app(transport="sse")
        logger.info("Token for Auth: %s", token)
        uvicorn.run(sse_app, host="0.0.0.0", port=8000)
    else:
        sse_app = mcp.http_app(transport="streamable-http")
        logger.info("Token for Auth: %s", token)
        uvicorn.run(sse_app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
