import argparse
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional, List
from dotenv import load_dotenv
from fastmcp import FastMCP, Context
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

ef = BGEM3EmbeddingFunction(use_fp16=False, device="gcu")


class MilvusConnector:

    def __init__(
        self,
        uri: str,
        token: Optional[str] = None,
        db_name: Optional[str] = "default",
        criminal_case_collection_name: Optional[str] = "criminal_case",
        criminal_case_parent_collection_name: Optional[
            str] = "criminal_case_parent",
        civil_case_collection_name: Optional[str] = "civil_case",
        civil_case_parent_collection_name: Optional[str] = "civil_case_parent"
    ):
        self.uri = uri
        self.token = token
        connections.connect(uri=uri, token=token, db_name=db_name)
        self.criminal_case_collection = Collection(
            criminal_case_collection_name)
        self.criminal_case_parent_collection = Collection(
            criminal_case_parent_collection_name)
        self.civil_case_collection = Collection(civil_case_collection_name)
        self.civil_case_parent_collection = Collection(
            civil_case_parent_collection_name)
        self.check_collections()
        self.criminal_case_collection.load()
        self.criminal_case_parent_collection.load()
        self.civil_case_collection.load()
        self.civil_case_parent_collection.load()

        self.criminal_output_fields = [
            "chunk", "relevant_articles", "accusation", "punish_of_money",
            "criminals", "imprisonment", "life_imprisonment", "death_penalty",
            "dates", "locations", "people", "numbers", "criminals_llm"
        ]
        self.civil_output_fields = [
            "chunk", "case_number", "case_name", "court", "region",
            "judgment_date", "parties", "case_cause", "legal_basis",
            "parent_id", "dates", "locations", "people", "numbers",
            "parties_llm"
        ]

    def check_collections(self) -> None:
        """Check if collections have the expected schema."""

        def check_fields(expected_fields, actual_fields, schema):
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

        criminal_expected_fields = [
            {
                'name': 'chunk_id',
                'type': 21
            },  # VARCHAR
            {
                'name': 'parent_id',
                'type': 21
            },  # VARCHAR
            {
                'name': 'fact_id',
                'type': 21
            },  # VARCHAR
            {
                'name': 'chunk',
                'type': 21
            },  # VARCHAR
            {
                'name': 'relevant_articles',
                'type': 22
            },  # ARRAY
            {
                'name': 'accusation',
                'type': 21
            },  # VARCHAR
            {
                'name': 'punish_of_money',
                'type': 5
            },  # INT64
            {
                'name': 'criminals',
                'type': 21
            },  # VARCHAR
            {
                'name': 'imprisonment',
                'type': 5
            },  # INT64
            {
                'name': 'life_imprisonment',
                'type': 1
            },  # BOOLEAN
            {
                'name': 'death_penalty',
                'type': 1
            },  # BOOLEAN
            {
                'name': 'dates',
                'type': 21
            },  # VARCHAR
            {
                'name': 'locations',
                'type': 21
            },  # VARCHAR
            {
                'name': 'people',
                'type': 21
            },  # VARCHAR
            {
                'name': 'numbers',
                'type': 21
            },  # VARCHAR
            {
                'name': 'criminals_llm',
                'type': 21
            },  # VARCHAR
            {
                'name': 'sparse_vector',
                'type': 104
            },  # SPARSE_FLOAT_VECTOR
            {
                'name': 'dense_vector',
                'type': 101
            },  # FLOAT_VECTOR
        ]

        criminal_schema = self.criminal_case_collection.describe()
        criminal_actual_fields = criminal_schema['fields']
        check_fields(criminal_expected_fields, criminal_actual_fields,
                     criminal_schema)

        criminal_parent_expected_fields = [
            {
                'name': 'parent_id',
                'type': 21
            },  # VARCHAR
            {
                'name': 'fact_id',
                'type': 21
            },  # VARCHAR
            {
                'name': 'chunk',
                'type': 21
            },  # VARCHAR
        ]

        criminal_parent_schema = self.criminal_case_parent_collection.describe(
        )
        criminal_parent_actual_fields = criminal_parent_schema['fields']
        check_fields(criminal_parent_expected_fields,
                     criminal_parent_actual_fields, criminal_parent_schema)

        civil_expected_fields = [
            {
                'name': 'chunk_id',
                'type': 21
            },  # VARCHAR
            {
                'name': 'case_id',
                'type': 21
            },  # VARCHAR
            {
                'name': 'chunk',
                'type': 21
            },  # VARCHAR
            {
                'name': 'case_number',
                'type': 21
            },  # VARCHAR
            {
                'name': 'case_name',
                'type': 21
            },  # VARCHAR
            {
                'name': 'court',
                'type': 21
            },  # VARCHAR
            {
                'name': 'region',
                'type': 21
            },  # VARCHAR
            {
                'name': 'judgment_date',
                'type': 21
            },  # VARCHAR
            {
                'name': 'parties',
                'type': 21
            },  # VARCHAR
            {
                'name': 'case_cause',
                'type': 21
            },  # VARCHAR
            {
                'name': 'legal_basis',
                'type': 23
            },  # JSON
            {
                'name': 'parent_id',
                'type': 21
            },  # VARCHAR
            {
                'name': 'dates',
                'type': 21
            },  # VARCHAR
            {
                'name': 'locations',
                'type': 21
            },  # VARCHAR
            {
                'name': 'people',
                'type': 21
            },  # VARCHAR
            {
                'name': 'numbers',
                'type': 21
            },  # VARCHAR
            {
                'name': 'parties_llm',
                'type': 21
            },  # VARCHAR
            {
                'name': 'sparse_vector',
                'type': 104
            },  # SPARSE_FLOAT_VECTOR
            {
                'name': 'dense_vector',
                'type': 101
            },  # FLOAT_VECTOR
        ]

        civil_schema = self.civil_case_collection.describe()
        civil_actual_fields = civil_schema['fields']
        check_fields(civil_expected_fields, civil_actual_fields, civil_schema)

        civil_parent_expected_fields = [
            {
                'name': 'parent_id',
                'type': 21
            },  # VARCHAR
            {
                'name': 'case_id',
                'type': 21
            },  # VARCHAR
            {
                'name': 'chunk',
                'type': 21
            },  # VARCHAR
        ]

        civil_parent_schema = self.civil_case_parent_collection.describe()
        civil_parent_actual_fields = civil_parent_schema['fields']
        check_fields(civil_parent_expected_fields, civil_parent_actual_fields,
                     civil_parent_schema)

    async def sparse_search(self,
                            collection: Collection,
                            query_text: str,
                            limit: int = 5,
                            filter_expr: Optional[str] = None,
                            parent_child: bool = False) -> list[dict]:
        """
        Perform sparse vector search on a collection.

        Args:
            collection: Collection to search
            query_text: Text to search for
            limit: Maximum number of results
            filter_expr: Optional filter expression for metadata filtering
            parent_child: Whether to use Parent Document Retrieval
        """
        if collection == self.civil_case_collection:
            output_fields = self.civil_output_fields
            parent_collection = self.civil_case_parent_collection
        else:
            output_fields = self.criminal_output_fields
            parent_collection = self.criminal_case_parent_collection

        try:
            query_embeddings = ef([query_text])
            sparse_embedding = query_embeddings["sparse"][[0]]

            sparse_search_params = {"metric_type": "IP", "params": {}}
            results = collection.search(data=[sparse_embedding],
                                        anns_field="sparse_vector",
                                        param=sparse_search_params,
                                        limit=limit,
                                        filter=filter_expr,
                                        output_fields=output_fields)[0]

            if parent_child:
                parent_ids = [result.parent_id for result in results]
                expr = f'parent_id IN {parent_ids}'
                parent_results = parent_collection.query(
                    expr=expr, output_fields=["parent_id", "chunk"])
                parent_chunks = {
                    parent["parent_id"]: parent["chunk"]
                    for parent in parent_results
                }
                for result in results:
                    result["parent_chunk"] = parent_chunks.get(
                        result.parent_id)

            return results
        except Exception as e:
            raise ValueError(f"Search failed: {str(e)}")

    async def dense_search(self,
                           collection: Collection,
                           query_text: str,
                           limit: int = 5,
                           filter_expr: Optional[str] = None,
                           parent_child: bool = False) -> list[dict]:
        """
        Perform dense vector search on a collection.

        Args:
            collection: Collection to search
            query_text: Text to search for
            limit: Maximum number of results
            filter_expr: Optional filter expression for metadata filtering
            parent_child: Whether to use Parent Document Retrieval
        """
        if collection == self.criminal_case_collection:
            output_fields = self.criminal_output_fields
            parent_collection = self.criminal_case_parent_collection
        else:
            output_fields = self.civil_output_fields
            parent_collection = self.civil_case_parent_collection

        try:
            query_embeddings = ef([query_text])
            dense_embedding = query_embeddings["dense"][0]

            dense_search_params = {"metric_type": "COSINE", "params": {}}
            results = collection.search(data=[dense_embedding],
                                        anns_field="dense_vector",
                                        param=dense_search_params,
                                        limit=limit,
                                        filter=filter_expr,
                                        output_fields=output_fields)[0]

            if parent_child:
                parent_ids = [result.parent_id for result in results]
                expr = f'parent_id IN {parent_ids}'
                parent_results = parent_collection.query(
                    expr=expr, output_fields=["parent_id", "chunk"])
                parent_chunks = {
                    parent["parent_id"]: parent["chunk"]
                    for parent in parent_results
                }
                for result in results:
                    result["parent_chunk"] = parent_chunks.get(
                        result.parent_id)

            return results
        except Exception as e:
            raise ValueError(f"Vector search failed: {str(e)}")

    async def hybrid_search(self,
                            collection: Collection,
                            query_text: str,
                            limit: int = 5,
                            filter_expr: Optional[str] = None,
                            parent_child: bool = False) -> list[dict]:
        """
        Perform hybrid search combining sparse vector search and dense vector search with RRF ranking.

        Args:
            collection: Collection to search
            query_text: Text to search for
            limit: Maximum number of results
            filter_expr: Optional filter expression for metadata filtering
            parent_child: Whether to use Parent Document Retrieval
        """
        if collection == self.civil_case_collection:
            output_fields = self.civil_output_fields
            parent_collection = self.civil_case_parent_collection
        else:
            output_fields = self.criminal_output_fields
            parent_collection = self.criminal_case_parent_collection

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
            results = collection.hybrid_search([sparse_request, dense_request],
                                               rerank=RRFRanker(60),
                                               limit=limit,
                                               output_fields=output_fields)[0]

            if parent_child:
                parent_ids = [result.parent_id for result in results]
                expr = f'parent_id IN {parent_ids}'
                parent_results = parent_collection.query(
                    expr=expr, output_fields=["parent_id", "chunk"])
                parent_chunks = {
                    parent["parent_id"]: parent["chunk"]
                    for parent in parent_results
                }
                for result in results:
                    result["parent_chunk"] = parent_chunks.get(
                        result.parent_id)

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
        criminal_case_collection_name=config.get(
            "criminal_case_collection_name", "criminal_case"),
        criminal_case_parent_collection_name=config.get(
            "criminal_case_parent_collection_name", "criminal_case_parent"),
        civil_case_collection_name=config.get("civil_case_collection_name",
                                              "civil_case"),
        civil_case_parent_collection_name=config.get(
            "civil_case_parent_collection_name", "civil_case_parent"),
    )

    try:
        yield MilvusContext(connector)
    finally:
        pass


mcp = FastMCP(name="Milvus", lifespan=server_lifespan)

@mcp.tool()
async def criminal_case_sparse_search(
    query_text: str,
    limit: int = 10,
    filter_expr: Optional[str] = None,
    parent_child: bool = False,
    ctx: Context = None,
) -> str:
    """
    稀疏检索刑事案例 Collection。适用于专有名词、人名、地名等。

    Collection 字段（元数据）说明：

    - relevant_articles (ARRAY[INT64]): 刑法相关法条
        示例：'ARRAY_CONTAINS(relevant_articles, 234)', 'ARRAY_CONTAINS_ALL(relevant_articles, [263, 264])', 'ARRAY_CONTAINS_ANY(relevant_articles, [263, 264])'
    - accusation (VARCHAR): 罪名
    - punish_of_money (INT64): 罚金
    - criminals (VARCHAR): 犯罪人
    - imprisonment (INT64): 刑期
    - life_imprisonment (BOOL): 无期徒刑
        示例：'life_imprisonment'
    - death_penalty (BOOL): 死刑
    - dates (VARCHAR): 日期
        示例：'dates like "%20210101%"'
    - locations (VARCHAR): 地点
    - people (VARCHAR): 人物（人物的姓名、职业和在当次审判中的角色）
    - numbers (VARCHAR): 数额
        示例：'numbers like "%16.8万元%"'
    - criminals_llm (VARCHAR): 犯罪人（通过 LLM 提取）
    
    注意：

    1. parent_child 参数默认取 False，当需要参考更完整的上下文时，设为 True 以返回父文档的 chunk。

    Args:
        query_text: Text to search for
        limit: Maximum number of results to return
        filter_expr: Optional filter expression for metadata filtering, e.g. 'accusation like "%毒品%" and imprisonment > 120', 'death_penalty'
        parent_child: Whether to use Parent Document Retrieval
    """
    connector = ctx.request_context.lifespan_context.connector
    results = await connector.sparse_search(
        collection=connector.criminal_case_collection,
        query_text=query_text,
        limit=limit,
        filter_expr=filter_expr,
        parent_child=parent_child)

    output = f"Sparse vector search results for '{query_text}':\n\n"
    for result in results:
        output += f"{result}\n\n"

    return output

@mcp.tool()
async def criminal_case_dense_search(
    query_text: str,
    limit: int = 10,
    filter_expr: Optional[str] = None,
    parent_child: bool = False,
    ctx: Context = None,
) -> str:
    """
    密集检索刑事案例 Collection。适用于基于语义的检索。

    Collection 字段（元数据）说明：

    - relevant_articles (ARRAY[INT64]): 刑法相关法条
        示例：'ARRAY_CONTAINS(relevant_articles, 234)', 'ARRAY_CONTAINS_ALL(relevant_articles, [263, 264])', 'ARRAY_CONTAINS_ANY(relevant_articles, [263, 264])'
    - accusation (VARCHAR): 罪名
    - punish_of_money (INT64): 罚金
    - criminals (VARCHAR): 犯罪人
    - imprisonment (INT64): 刑期
    - life_imprisonment (BOOL): 无期徒刑
        示例：'life_imprisonment'
    - death_penalty (BOOL): 死刑
    - dates (VARCHAR): 日期
        示例：'dates like "%20210101%"'
    - locations (VARCHAR): 地点
    - people (VARCHAR): 人物（人物的姓名、职业和在当次审判中的角色）
    - numbers (VARCHAR): 数额
        示例：'numbers like "%16.8万元%"'
    - criminals_llm (VARCHAR): 犯罪人（通过 LLM 提取）
    
    注意：

    1. parent_child 参数默认取 False，当需要参考更完整的上下文时，设为 True 以返回父文档的 chunk。

    Args:
        query_text: Text to search for
        limit: Maximum number of results to return
        filter_expr: Optional filter expression for metadata filtering, e.g. 'accusation like "%毒品%" and imprisonment > 120', 'death_penalty'
        parent_child: Whether to use Parent Document Retrieval
    """
    connector = ctx.request_context.lifespan_context.connector
    results = await connector.dense_search(
        collection=connector.criminal_case_collection,
        query_text=query_text,
        limit=limit,
        filter_expr=filter_expr,
        parent_child=parent_child)

    output = f"Dense vector search results for '{query_text}':\n\n"
    for result in results:
        output += f"{result}\n\n"

    return output

@mcp.tool()
async def criminal_case_hybrid_search(
    query_text: str,
    limit: int = 10,
    filter_expr: Optional[str] = None,
    parent_child: bool = False,
    ctx: Context = None,
) -> str:
    """
    混合检索刑事案例 Collection。适用于大多数情况。

    Collection 字段（元数据）说明：

    - relevant_articles (ARRAY[INT64]): 刑法相关法条
        示例：'ARRAY_CONTAINS(relevant_articles, 234)', 'ARRAY_CONTAINS_ALL(relevant_articles, [263, 264])', 'ARRAY_CONTAINS_ANY(relevant_articles, [263, 264])'
    - accusation (VARCHAR): 罪名
    - punish_of_money (INT64): 罚金
    - criminals (VARCHAR): 犯罪人
    - imprisonment (INT64): 刑期
    - life_imprisonment (BOOL): 无期徒刑
        示例：'life_imprisonment'
    - death_penalty (BOOL): 死刑
    - dates (VARCHAR): 日期
        示例：'dates like "%20210101%"'
    - locations (VARCHAR): 地点
    - people (VARCHAR): 人物（人物的姓名、职业和在当次审判中的角色）
    - numbers (VARCHAR): 数额
        示例：'numbers like "%16.8万元%"'
    - criminals_llm (VARCHAR): 犯罪人（通过 LLM 提取）
    
    注意：

    1. parent_child 参数默认取 False，当需要参考更完整的上下文时，设为 True 以返回父文档的 chunk。

    Args:
        query_text: Text to search for
        limit: Maximum number of results to return
        filter_expr: Optional filter expression for metadata filtering, e.g. 'accusation like "%毒品%" and imprisonment > 120', 'death_penalty'
        parent_child: Whether to use Parent Document Retrieval
    """
    connector = ctx.request_context.lifespan_context.connector

    results = await connector.hybrid_search(
        collection=connector.criminal_case_collection,
        query_text=query_text,
        limit=limit,
        filter_expr=filter_expr,
        parent_child=parent_child,
    )

    output = f"Hybrid search results for text '{query_text}':\n\n"
    for result in results:
        output += f"{result}\n\n"

    return output


@mcp.tool()
async def civil_case_sparse_search(
    query_text: str,
    limit: int = 10,
    filter_expr: Optional[str] = None,
    parent_child: bool = False,
    ctx: Context = None,
) -> str:
    """
    稀疏检索民事案例 Collection。适用于专有名词、人名、地名等。

    Collection 字段（元数据）说明：

    - case_number (VARCHAR): 案号
    - case_name (VARCHAR): 案件名称
    - court (VARCHAR): 法院
    - region (VARCHAR): 所属地区
    - judgment_date (VARCHAR): 判决日期
        示例：'judgment_date like "%2021年1月1日%"'
    - parties (VARCHAR): 当事人
    - case_cause (VARCHAR): 案由
    - legal_basis (JSON): 法律依据
        示例：'json_contains(legal_basis["中华人民共和国民法典"], 60)', 'json_contains_all(legal_basis["中华人民共和国民法典"], [107, 109])', 'json_contains_any(legal_basis["中华人民共和国民法典"], [107, 109])'
    - dates (VARCHAR): 日期
        示例：'dates like "%20210101%"'
    - locations (VARCHAR): 地点
    - people (VARCHAR): 人物（人物的姓名、职业和在当次审判中的角色）
    - numbers (VARCHAR): 数额（通过 LLM 提取）
        示例：'numbers like "%16.8万元%"'
    - parties_llm (VARCHAR): 当事人（通过 LLM 提取）
    
    注意：

    1. parent_child 参数默认取 False，当需要参考更完整的上下文时，设为 True 以返回父文档的 chunk。

    Args:
        query_text: Text to search for
        limit: Maximum number of results to return.
        filter_expr: Optional filter expression for metadata filtering, e.g. 'court like "%沈阳市%" and judgment_date like "%2021年9月%"', 'case_cause == "合同纠纷"'
        parent_child: Whether to use Parent Document Retrieval
    """
    connector = ctx.request_context.lifespan_context.connector
    results = await connector.sparse_search(
        collection=connector.civil_case_collection,
        query_text=query_text,
        limit=limit,
        filter_expr=filter_expr,
        parent_child=parent_child)

    output = f"Sparse vector search results for '{query_text}':\n\n"
    for result in results:
        output += f"{result}\n\n"

    return output


@mcp.tool()
async def civil_case_dense_search(
    query_text: str,
    limit: int = 10,
    filter_expr: Optional[str] = None,
    parent_child: bool = False,
    ctx: Context = None,
) -> str:
    """
    密集检索民事案例 Collection。适用于基于语义的检索。

    Collection 字段（元数据）说明：

    - case_number (VARCHAR): 案号
    - case_name (VARCHAR): 案件名称
    - court (VARCHAR): 法院
    - region (VARCHAR): 所属地区
    - judgment_date (VARCHAR): 判决日期
        示例：'judgment_date like "%2021年1月1日%"'
    - parties (VARCHAR): 当事人
    - case_cause (VARCHAR): 案由
    - legal_basis (JSON): 法律依据
        示例：'json_contains(legal_basis["中华人民共和国民法典"], 60)', 'json_contains_all(legal_basis["中华人民共和国民法典"], [107, 109])', 'json_contains_any(legal_basis["中华人民共和国民法典"], [107, 109])'
    - dates (VARCHAR): 日期
        示例：'dates like "%20210101%"'
    - locations (VARCHAR): 地点
    - people (VARCHAR): 人物（人物的姓名、职业和在当次审判中的角色）
    - numbers (VARCHAR): 数额（通过 LLM 提取）
        示例：'numbers like "%16.8万元%"'
    - parties_llm (VARCHAR): 当事人（通过 LLM 提取）
    
    注意：

    1. parent_child 参数默认取 False，当需要参考更完整的上下文时，设为 True 以返回父文档的 chunk。

    Args:
        query_text: Text to search for
        limit: Maximum number of results to return.
        filter_expr: Optional filter expression for metadata filtering, e.g. 'court like "%沈阳市%" and judgment_date like "%2021年9月%"', 'case_cause == "合同纠纷"'
        parent_child: Whether to use Parent Document Retrieval
    """
    connector = ctx.request_context.lifespan_context.connector
    results = await connector.dense_search(
        collection=connector.civil_case_collection,
        query_text=query_text,
        limit=limit,
        filter_expr=filter_expr,
        parent_child=parent_child)

    output = f"Dense vector search results for '{query_text}':\n\n"
    for result in results:
        output += f"{result}\n\n"

    return output


@mcp.tool()
async def civil_case_hybrid_search(
    query_text: str,
    limit: int = 10,
    filter_expr: Optional[str] = None,
    parent_child: bool = False,
    ctx: Context = None,
) -> str:
    """
    混合检索民事案例 Collection。适用于大多数情况。

    Collection 字段（元数据）说明：

    - case_number (VARCHAR): 案号
    - case_name (VARCHAR): 案件名称
    - court (VARCHAR): 法院
    - region (VARCHAR): 所属地区
    - judgment_date (VARCHAR): 判决日期
        示例：'judgment_date like "%2021年1月1日%"'
    - parties (VARCHAR): 当事人
    - case_cause (VARCHAR): 案由
    - legal_basis (JSON): 法律依据
        示例：'json_contains(legal_basis["中华人民共和国民法典"], 60)', 'json_contains_all(legal_basis["中华人民共和国民法典"], [107, 109])', 'json_contains_any(legal_basis["中华人民共和国民法典"], [107, 109])'
    - dates (VARCHAR): 日期
        示例：'dates like "%20210101%"'
    - locations (VARCHAR): 地点
    - people (VARCHAR): 人物（人物的姓名、职业和在当次审判中的角色）
    - numbers (VARCHAR): 数额（通过 LLM 提取）
        示例：'numbers like "%16.8万元%"'
    - parties_llm (VARCHAR): 当事人（通过 LLM 提取）
    
    注意：

    1. parent_child 参数默认取 False，当需要参考更完整的上下文时，设为 True 以返回父文档的 chunk。

    Args:
        query_text: Text to search for
        limit: Maximum number of results to return.
        filter_expr: Optional filter expression for metadata filtering, e.g. 'court like "%沈阳市%" and judgment_date like "%2021年9月%"', 'case_cause == "合同纠纷"'
        parent_child: Whether to use Parent Document Retrieval
    """
    connector = ctx.request_context.lifespan_context.connector

    results = await connector.hybrid_search(
        collection=connector.civil_case_collection,
        query_text=query_text,
        limit=limit,
        filter_expr=filter_expr,
        parent_child=parent_child,
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
        "criminal_case_collection_name":
        os.environ.get("MILVUS_COLLECTION_CRIMINAL_CASE"),
        "criminal_case_parent_collection_name":
        os.environ.get("MILVUS_COLLECTION_CRIMINAL_CASE_PARENT"),
        "civil_case_collection_name":
        os.environ.get("MILVUS_COLLECTION_CIVIL_CASE"),
        "civil_case_parent_collection_name":
        os.environ.get("MILVUS_COLLECTION_CIVIL_CASE_PARENT"),
    }

    if args.sse:
        mcp.run(transport="sse", port=8000, host="0.0.0.0")
    else:
        mcp.run(transport="streamable-http", port=8000, host="0.0.0.0")


if __name__ == "__main__":
    main()
