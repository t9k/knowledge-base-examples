# 检索法律案例知识库的 MCP Server

这是一个基于 MCP (Model Context Protocol) 和 Milvus 的法律案例检索服务，专门用于检索中国刑事案例和民事案例。该服务使用 Qwen3-Embedding-0.6B 和 BGE-M3 模型提供多种检索方式，支持元数据过滤和向量检索，支持父子文档检索以获得更完整的上下文。

## 功能

- 实现了完整的 MCP 服务器协议，可与支持 MCP 的客户端集成
- 支持中国刑事案例和民事案例的检索
- Qwen3-Embedding-0.6B 模型提供密集嵌入，BGE-M3 模型提供稀疏嵌入，支持四种检索模式：
  - `query`: 使用过滤表达式进行精确查询
  - `sparse_search`: 稀疏向量检索，适用于专有名词、人名、地名等
  - `dense_search`: 密集向量检索，适用于语义检索
  - `hybrid_search`: 混合检索，结合稀疏和密集向量
- 支持元数据筛选和条件查询
- 支持父子文档检索模式，可获得更完整的案例上下文
- 提供健康检查端点
- 支持 sse 和 streamable-http 传输模式
- 可配置的身份认证（JWT Bearer Token）

### 身份认证

当 `ENABLE_AUTH=true` 时，服务器会启用JWT Bearer Token认证：
- 服务启动时会动态生成RSA密钥对
- 生成30天有效期的JWT token
- 客户端需要在请求头中包含 `Authorization: Bearer <token>`

当 `ENABLE_AUTH=false` 时，服务器不需要任何认证即可访问。

## 部署

### 在 Kubernetes 部署

修改 `k8s.yaml` 中的 ConfigMap 和 VirtualService 配置，然后执行以下命令进行部署：

```bash
kubectl apply -f k8s.yaml
```

### 在本地部署

1. 安装依赖：

```bash
pip install pymilvus[model]==2.5.10 fastmcp==2.8.1 python-dotenv uvicorn
```

2. 创建 `.env` 文件：

```env
MILVUS_URI=http://localhost:19530
MILVUS_TOKEN=your_token
MILVUS_DB=default
MILVUS_COLLECTION_CRIMINAL_CASE=criminal_case
MILVUS_COLLECTION_CRIMINAL_CASE_PARENT=criminal_case_parent
MILVUS_COLLECTION_CIVIL_CASE=civil_case
MILVUS_COLLECTION_CIVIL_CASE_PARENT=civil_case_parent
EMBEDDING_BASE_URL=http://app-vllm-enflame-xxxxxxxx.demo.ksvc.qy.t9kcloud.cn/v1
EMBEDDING_MODEL=Qwen3-Embedding-0.6B
ENABLE_AUTH=false
```

3. 启动服务：

```bash
# 启动 sse 模式
python server.py --sse

# 启动 streamable-http 模式
python server.py
```

## 混合检索支持

服务调用 Qwen3-Embedding-0.6B 模型提供密集嵌入，集成 BGE-M3 模型提供稀疏嵌入，支持同时使用密集向量和稀疏向量进行混合检索，通过 RRF (Reciprocal Rank Fusion) 算法融合两种检索结果，提高检索质量。

### 混合检索工作流程

1. 使用 Qwen3-Embedding-0.6B 和 BGE-M3 模型将用户查询转换为密集向量和稀疏向量
2. 分别用密集向量和稀疏向量在 Milvus 中进行检索
3. 使用 RRF 算法对两种检索结果进行融合排序
4. 返回包含元数据的检索结果

## 父子文档检索

服务支持父子文档检索模式（Parent Document Retrieval），通过设置 `parent_child=True` 参数：

- 子文档用于向量检索和匹配
- 返回对应的父文档内容，提供更完整的案例上下文
- 适用于需要完整案例信息的场景

## Milvus Collection 要求

### 刑事案例 Collection 字段

刑事案例 Collection (`criminal_case`) 需要包含以下字段：

- `chunk_id` (VarChar)：chunk 的唯一标识符
- `parent_id` (VarChar)：父文档标识符
- `fact_id` (VarChar)：案例事实标识符
- `chunk` (VarChar)：案例内容片段
- `relevant_articles` (Array[Int64])：相关刑法法条
- `accusation` (VarChar)：罪名
- `punish_of_money` (Int64)：罚金（单位：元）
- `criminals` (VarChar)：犯罪人
- `imprisonment` (Int64)：刑期（单位：月）
- `life_imprisonment` (Boolean)：是否无期徒刑
- `death_penalty` (Boolean)：是否死刑
- `dates` (VarChar)：提取的日期信息
- `locations` (VarChar)：提取的地点信息
- `people` (VarChar)：提取的人物信息
- `numbers` (VarChar)：提取的数额信息
- `criminals_llm` (VarChar)：通过 LLM 提取的犯罪人信息
- `sparse_vector` (SparseFloatVector)：稀疏嵌入向量
- `dense_vector` (FloatVector)：密集嵌入向量

### 刑事案例父文档 Collection 字段

刑事案例父文档 Collection (`criminal_case_parent`) 需要包含以下字段：

- `parent_id` (VarChar)：父文档标识符
- `fact_id` (VarChar)：案例事实标识符
- `chunk` (VarChar)：完整的案例内容

### 民事案例 Collection 字段

民事案例 Collection (`civil_case`) 需要包含以下字段：

- `chunk_id` (VarChar)：chunk 的唯一标识符
- `case_id` (VarChar)：案例标识符
- `chunk` (VarChar)：案例内容片段
- `case_number` (VarChar)：案号
- `case_name` (VarChar)：案件名称
- `court` (VarChar)：审理法院
- `region` (VarChar)：所属地区
- `judgment_date` (VarChar)：判决日期
- `parties` (VarChar)：当事人
- `case_cause` (VarChar)：案由
- `legal_basis` (JSON)：法律依据
- `parent_id` (VarChar)：父文档标识符
- `dates` (VarChar)：提取的日期信息
- `locations` (VarChar)：提取的地点信息
- `people` (VarChar)：提取的人物信息
- `numbers` (VarChar)：提取的数额信息
- `parties_llm` (VarChar)：通过 LLM 提取的当事人信息
- `sparse_vector` (SparseFloatVector)：稀疏嵌入向量
- `dense_vector` (FloatVector)：密集嵌入向量

### 民事案例父文档 Collection 字段

民事案例父文档 Collection (`civil_case_parent`) 需要包含以下字段：

- `parent_id` (VarChar)：父文档标识符
- `case_id` (VarChar)：案例标识符
- `chunk` (VarChar)：完整的案例内容

## MCP 工具列表

刑事案例检索：

1. `criminal_case_query`：使用过滤表达式进行精确查询
1. `criminal_case_sparse_search`：稀疏向量检索，适用于专有名词、人名、地名等
1. `criminal_case_dense_search`：密集向量检索，适用于语义检索
1. `criminal_case_hybrid_search`：混合检索，适用于大多数情况

民事案例检索：

1. `civil_case_query`：使用过滤表达式进行精确查询
1. `civil_case_sparse_search`：稀疏向量检索，适用于专有名词、人名、地名等
1. `civil_case_dense_search`：密集向量检索，适用于语义检索
1. `civil_case_hybrid_search`：混合检索，适用于大多数情况
