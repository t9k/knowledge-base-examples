import argparse
import os
from contextlib import asynccontextmanager
import logging
from typing import Annotated, AsyncIterator, Optional
from pydantic import Field
from dotenv import load_dotenv
from fastmcp import FastMCP, Context
from fastmcp.server.auth import BearerAuthProvider
from fastmcp.server.auth.providers.bearer import RSAKeyPair
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from pymilvus import (
    connections,
    Collection,
    AnnSearchRequest,
    RRFRanker,
)
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from openai import OpenAI
import torch
import torch_gcu

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MilvusConnector:

    def __init__(self, uri: str, token: str, db_name: str,
                 criminal_law_collection_name: str,
                 civil_code_collection_name: str, embedding_base_url: str,
                 embedding_model: str):
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

        self.embedding_client = OpenAI(base_url=embedding_base_url,
                                       api_key="dummy")
        self.embedding_model = embedding_model

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
            sparse_embedding = ef([query_text])["sparse"][[0]]

            sparse_search_params = {"metric_type": "IP", "params": {}}
            results = collection.search(data=[sparse_embedding],
                                        anns_field="sparse_vector",
                                        param=sparse_search_params,
                                        limit=limit,
                                        expr=filter_expr,
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
            dense_embedding = self.embedding_client.embeddings.create(
                input=[query_text],
                model=self.embedding_model).data[0].embedding

            dense_search_params = {"metric_type": "COSINE", "params": {}}
            results = collection.search(data=[dense_embedding],
                                        anns_field="dense_vector",
                                        param=dense_search_params,
                                        limit=limit,
                                        expr=filter_expr,
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
            sparse_embedding = ef([query_text])["sparse"][[0]]
            dense_embedding = self.embedding_client.embeddings.create(
                input=[query_text],
                model=self.embedding_model).data[0].embedding

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
        embedding_base_url=config.get("embedding_base_url"),
        embedding_model=config.get("embedding_model"),
    )

    try:
        yield MilvusContext(connector)
    finally:
        pass


enable_auth = os.environ.get("ENABLE_AUTH", "false") == "true"
if enable_auth:
    # 动态生成RSA密钥对
    key_pair = RSAKeyPair.generate()
    auth = BearerAuthProvider(public_key=key_pair.public_key)
    token = key_pair.create_token(expires_in_seconds=86400 * 30)
    mcp = FastMCP(name="Law Searcher", lifespan=server_lifespan, auth=auth)
    logger.info("Authentication enabled")
else:
    mcp = FastMCP(name="Law Searcher", lifespan=server_lifespan)
    logger.info("Authentication disabled")


@mcp.resource("data://criminal-law-contents")
async def criminal_law_contents() -> str:
    """民法典的目录，用于参考。"""

    return """第一编 总则  
  第一章 刑法的任务、基本原则和适用范围  
  第二章 犯罪  
    第一节 犯罪和刑事责任  
    第二节 犯罪的预备、未遂和中止  
    第三节 共同犯罪  
    第四节 单位犯罪  
  第三章 刑罚  
    第一节 刑罚的种类  
    第二节 管制  
    第三节 拘役  
    第四节 有期徒刑、无期徒刑  
    第五节 死刑  
    第六节 罚金  
    第七节 剥夺政治权利  
    第八节 没收财产  
  第四章 刑罚的具体运用  
    第一节 量刑  
    第二节 累犯  
    第三节 自首和立功  
    第四节 数罪并罚  
    第五节 缓刑  
    第六节 减刑  
    第七节 假释  
    第八节 时效  
  第五章 其他规定  

第二编 分则  
  第一章 危害国家安全罪  
  第二章 危害公共安全罪  
  第三章 破坏社会主义市场经济秩序罪  
    第一节 生产、销售伪劣商品罪  
    第二节 走私罪  
    第三节 妨害对公司、企业的管理秩序罪  
    第四节 破坏金融管理秩序罪  
    第五节 金融诈骗罪  
    第六节 危害税收征管罪  
    第七节 侵犯知识产权罪  
    第八节 扰乱市场秩序罪  
  第四章 侵犯公民人身权利、民主权利罪  
  第第五章 侵犯财产罪  
  第六章 妨害社会管理秩序罪  
    第一节 扰乱公共秩序罪  
    第二节 妨害司法罪  
    第三节 妨害国（边）境管理罪  
    第四节 妨害文物管理罪  
    第五节 危害公共卫生罪  
    第六节 破坏环境资源保护罪  
    第七节 走私、贩卖、运输、制造毒品罪  
    第八节 组织、强迫、引诱、容留、介绍卖淫罪  
    第九节 制作、贩卖、传播淫秽物品罪  
  第七章 危害国防利益罪  
  第八章 贪污贿赂罪  
  第九章 渎职罪  
  第十章 军人违反职责罪附则  

附则
"""


@mcp.tool()
async def criminal_law_query(
    filter_expr: Annotated[
        str,
        Field(description=(
            "Filter expression, e.g. 'law == \"中华人民共和国刑法\" and "
            "article == 123', 'law like \"%十一%\" and article_amended == 338'"
        ))],
    limit: Annotated[
        int,
        Field(description="Maximum number of results to return", ge=1, le=100
              )] = 5,
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
    query_text: Annotated[str, Field(description="Text to search for")],
    limit: Annotated[
        int,
        Field(description="Maximum number of results to return", ge=1, le=100
              )] = 5,
    filter_expr: Annotated[
        Optional[str],
        Field(description=(
            "Optional filter expression for metadata filtering, e.g. "
            "'chapter like \"第三章%\"', 'section == \"走私罪\"'"))] = None,
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
    query_text: Annotated[str, Field(description="Text to search for")],
    limit: Annotated[
        int,
        Field(description="Maximum number of results to return", ge=1, le=100
              )] = 5,
    filter_expr: Annotated[
        Optional[str],
        Field(description=(
            "Optional filter expression for metadata filtering, e.g. "
            "'chapter like \"第三章%\"', 'section == \"走私罪\"'"))] = None,
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
    query_text: Annotated[str, Field(description="Text to search for")],
    limit: Annotated[
        int,
        Field(description="Maximum number of results to return", ge=1, le=100
              )] = 5,
    filter_expr: Annotated[
        Optional[str],
        Field(description=(
            "Optional filter expression for metadata filtering, e.g. "
            "'chapter like \"第三章%\"', 'section == \"走私罪\"'"))] = None,
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


@mcp.resource("data://civil-code-contents")
async def civil_code_contents() -> str:
    """民法典的目录，用于参考。"""

    return """第一编 总则
  第一章 基本规定
  第二章 自然人
    第一节 民事权利能力和民事行为能力
    第二节 监护
    第三节 宣告失踪和宣告死亡
    第四节 个体工商户和农村承包经营户
  第三章 法人
    第一节 一般规定
    第二节 营利法人
    第三节 非营利法人
    第四节 特别法人
  第四章 非法人组织
  第五章 民事权利
  第六章 民事法律行为
    第一节 一般规定
    第二节 意思表示
    第三节 民事法律行为的效力
    第四节 民事法律行为的附条件和附期限
  第七章 代理
    第一节 一般规定
    第二节 委托代理
    第三节 代理终止
  第八章 民事责任
  第九章 诉讼时效
  第十章 期间计算

第二编 物权
  第一分编 通则
    第一章 一般规定
    第二章 物权的设立、变更、转让和消灭
      第一节 不动产登记
      第二节 动产交付
      第三节 其他规定
    第三章 物权的保护
  第二分编 所有权
    第四章 一般规定
    第五章 国家所有权和集体所有权、私人所有权
    第六章 业主的建筑物区分所有权
    第七章 相邻关系
    第八章 共有
    第九章 所有权取得的特别规定
  第三分编 用益物权
    第十章 一般规定
    第十一章 土地承包经营权
    第十二章 建设用地使用权
    第十三章 宅基地使用权
    第十四章 居住权
    第十五章 地役权
  第四分编 担保物权
    第十六章 一般规定
    第十七章 抵押权
      第一节 一般抵押权
      第二节 最高额抵押权
    第十八章 质权
      第一节 动产质权
      第二节 权利质权
    第十九章 留置权
  第五分编 占有
    第二十章 占有

第三编 合同
  第一分编 通则
    第一章 一般规定
    第二章 合同的订立
    第三章 合同的效力
    第四章 合同的履行
    第五章 合同的保全
    第六章 合同的变更和转让
    第七章 合同的权利义务终止
    第八章 违约责任
  第二分编 典型合同
    第九章 买卖合同
    第十章 供用电、水、气、热力合同
    第十一章 赠与合同
    第十二章 借款合同
    第十三章 保证合同
      第一节 一般规定
      第二节 保证责任
    第十四章 租赁合同
    第十五章 融资租赁合同
    第十六章 保理合同
    第十七章 承揽合同
    第十八章 建设工程合同
    第十九章 运输合同
      第一节 一般规定
      第二节 客运合同
      第三节 货运合同
      第四节 多式联运合同
    第二十章 技术合同
      第一节 一般规定
      第二节 技术开发合同
      第三节 技术转让合同和技术许可合同
      第四节 技术咨询合同和技术服务合同
    第二十一章 保管合同
    第二十二章 仓储合同
    第二十三章 委托合同
    第二十四章 物业服务合同
    第二十五章 行纪合同
    第二十六章 中介合同
    第二十七章 合伙合同
  第三分编 准合同
    第二十八章 无因管理
    第二十九章 不当得利

第四编 人格权
  第一章 一般规定
  第二章 生命权、身体权和健康权
  第三章 姓名权和名称权
  第四章 肖像权
  第五章 名誉权和荣誉权
  第六章 隐私权和个人信息保护

第五编 婚姻家庭
  第一章 一般规定
  第二章 结婚
  第三章 家庭关系
    第一节 夫妻关系
    第二节 父母子女关系和其他近亲属关系
  第四章 离婚
  第五章 收养
    第一节 收养关系的成立
    第二节 收养的效力
    第三节 收养关系的解除

第六编 继承
  第一章 一般规定
  第二章 法定继承
  第三章 遗嘱继承和遗赠
  第四章 遗产的处理

第七编 侵权责任
  第一章 一般规定
  第二章 损害赔偿
  第三章 责任主体的特殊规定
  第四章 产品责任
  第五章 机动车交通事故责任
  第六章 医疗损害责任
  第七章 环境污染和生态破坏责任
  第八章 高度危险责任
  第九章 饲养动物损害责任
  第十章 建筑物和物件损害责任

附则
"""


@mcp.tool()
async def civil_code_query(
    filter_expr: Annotated[
        str,
        Field(description=(
            "Filter expression, e.g. 'part == \"第一编 总则\" and article == 123', "
            "'part like \"%第二编%第二分编%\" and article == 345'"))],
    limit: Annotated[
        int,
        Field(description="Maximum number of results to return", ge=1, le=100
              )] = 5,
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
    query_text: Annotated[str, Field(description="Text to search for")],
    limit: Annotated[
        int,
        Field(description="Maximum number of results to return", ge=1, le=100
              )] = 10,
    filter_expr: Annotated[
        Optional[str],
        Field(description=(
            "Optional filter expression for metadata filtering, e.g. "
            "'part like \"%婚姻家庭编%\"', 'chapter like \"%肖像权%\"'"))] = None,
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
    query_text: Annotated[str, Field(description="Text to search for")],
    limit: Annotated[
        int,
        Field(description="Maximum number of results to return", ge=1, le=100
              )] = 10,
    filter_expr: Annotated[
        Optional[str],
        Field(description=(
            "Optional filter expression for metadata filtering, e.g. "
            "'part like \"%婚姻家庭编%\"', 'chapter like \"%肖像权%\"'"))] = None,
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
    query_text: Annotated[str, Field(description="Text to search for")],
    limit: Annotated[
        int,
        Field(description="Maximum number of results to return", ge=1, le=100
              )] = 10,
    filter_expr: Annotated[
        Optional[str],
        Field(description=(
            "Optional filter expression for metadata filtering, e.g. "
            "'part like \"%婚姻家庭编%\"', 'chapter like \"%肖像权%\"'"))] = None,
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


if __name__ == "__main__":
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
        "embedding_base_url":
        os.environ.get("EMBEDDING_BASE_URL"),
        "embedding_model":
        os.environ.get("EMBEDDING_MODEL"),
    }

    ef = BGEM3EmbeddingFunction(use_fp16=False, device="gcu")

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
