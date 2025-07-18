# 检索法律条文知识库的 MCP Server

这是一个基于 MCP (Model Context Protocol) 和 Milvus 的法律条文检索服务，专门用于检索中国刑法和民法典条文。该服务使用 Qwen3-Embedding-0.6B 和 BGE-M3 模型提供多种检索方式，支持元数据过滤和向量检索。

## 功能

- 实现了完整的 MCP 服务器协议，可与支持 MCP 的客户端集成
- 支持中国刑法（包括十二部修正案）和民法典的条文检索
- Qwen3-Embedding-0.6B 模型提供密集嵌入，BGE-M3 模型提供稀疏嵌入，支持四种检索模式：
  - `query`: 使用过滤表达式进行精确查询
  - `sparse_search`: 稀疏向量检索，适用于专有名词
  - `dense_search`: 密集向量检索，适用于语义检索
  - `hybrid_search`: 混合检索，结合稀疏和密集向量
- 支持元数据筛选和条件查询
- 提供健康检查端点
- 支持 sse 和 streamable-http 传输模式
- 可配置的身份认证（JWT Bearer Token）

### 传输模式

#### HTTP 模式 (streamable-http)
- **特点**: 传统的 HTTP 请求/响应模式
- **适用场景**: 标准的 MCP 客户端集成
- **路径**: `/mcp/law-searcher/`

#### SSE 模式 (Server-Sent Events)
- **特点**: 实时双向流式通信
- **适用场景**: 需要实时交互的应用
- **路径**: `/mcp/law-searcher-sse/`
- **端点**: 
  - SSE 连接：`/mcp/law-searcher-sse/sse`
  - 消息发送：`/mcp/law-searcher-sse/message`

### 身份认证

当 `ENABLE_AUTH=true` 时，服务器会启用JWT Bearer Token认证：
- 服务启动时会动态生成RSA密钥对
- 生成30天有效期的JWT token
- 客户端需要在请求头中包含 `Authorization: Bearer <token>`

当 `ENABLE_AUTH=false` 时，服务器不需要任何认证即可访问。

## 部署

### 在 Kubernetes 部署

服务提供两种部署模式：

#### HTTP 模式部署
修改 `k8s.yaml` 中的 ConfigMap 和 VirtualService 配置，然后执行以下命令进行部署：

```bash
kubectl apply -f k8s.yaml
```

#### SSE 模式部署
修改 `k8s-sse.yaml` 中的 ConfigMap 和 VirtualService 配置，然后执行以下命令进行部署：

```bash
kubectl apply -f k8s-sse.yaml
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
MILVUS_COLLECTION_CRIMINAL_LAW=criminal_law
MILVUS_COLLECTION_CIVIL_CODE=civil_code
EMBEDDING_BASE_URL=http://app-vllm-enflame-xxxxxxxx.demo.ksvc.qy.t9kcloud.cn/v1
EMBEDDING_MODEL=Qwen3-Embedding-0.6B
ENABLE_AUTH=false
```

3. 启动服务：

```bash
# 启动 HTTP 模式 (streamable-http)
python server.py

# 启动 SSE 模式 (Server-Sent Events)
python server.py --sse
```

## 混合检索支持

服务调用 Qwen3-Embedding-0.6B 模型提供密集嵌入，集成 BGE-M3 模型提供稀疏嵌入，支持同时使用密集向量和稀疏向量进行混合检索，通过 RRF (Reciprocal Rank Fusion) 算法融合两种检索结果，提高检索质量。

### 混合检索工作流程

1. 使用 Qwen3-Embedding-0.6B 和 BGE-M3 模型将用户查询转换为密集向量和稀疏向量
2. 分别用密集向量和稀疏向量在 Milvus 中进行检索
3. 使用 RRF 算法对两种检索结果进行融合排序
4. 返回包含元数据的检索结果

## Milvus Collection 要求

### 刑法 Collection 字段

刑法 Collection (`criminal_law`) 需要包含以下字段：

- `chunk_id` (VarChar)：chunk 的唯一标识符
- `chunk` (VarChar)：法条内容
- `law` (VarChar)：所属法律名称（如"中华人民共和国刑法"、"中华人民共和国刑法修正案（二）"等）
- `part` (VarChar)：所属编
- `chapter` (VarChar)：所属章
- `section` (VarChar)：所属节
- `article` (Int64)：法条序号
- `article_amended` (Int64)：修正的刑法法条序号（仅修正案有此字段）
- `sparse_vector` (SparseFloatVector)：稀疏嵌入向量
- `dense_vector` (FloatVector)：密集嵌入向量

### 民法典 Collection 字段

民法典 Collection (`civil_code`) 需要包含以下字段：

- `chunk_id` (VarChar)：chunk 的唯一标识符
- `chunk` (VarChar)：法条内容
- `law` (VarChar)：法律名称（"中华人民共和国民法典"）
- `part` (VarChar)：所属编（如"第一编 总则"、"第二编 物权编 第一分编 通则"等）
- `chapter` (VarChar)：所属章
- `section` (VarChar)：所属节
- `article` (Int64)：法条序号
- `sparse_vector` (SparseFloatVector)：稀疏嵌入向量
- `dense_vector` (FloatVector)：密集嵌入向量

## MCP 工具列表

刑法检索：

1. `criminal_law_query`：使用过滤表达式进行精确查询
1. `criminal_law_sparse_search`：稀疏向量检索，适用于专有名词
1. `criminal_law_dense_search`：密集向量检索，适用于语义检索
1. `criminal_law_hybrid_search`：混合检索，适用于大多数情况

民法典检索：

1. `civil_code_query`：使用过滤表达式进行精确查询
1. `civil_code_sparse_search`：稀疏向量检索，适用于专有名词
1. `civil_code_dense_search`：密集向量检索，适用于语义检索
1. `civil_code_hybrid_search`：混合检索，适用于大多数情况
