# 检索法律条文知识库的 MCP Server

这是一个基于 MCP (Model Context Protocol) 和 Milvus 的法律条文检索服务，专门用于检索中国刑法和民法典条文。该服务使用 BGE-M3 模型提供多种检索方式，支持精确查询和语义检索。

## 功能

- 实现了完整的 MCP 服务器协议，可与支持 MCP 的客户端集成
- 支持中国刑法（包括十二部修正案）和民法典的条文检索
- 集成 BGE-M3 模型，支持四种检索模式：
  - `query`: 使用过滤表达式进行精确查询
  - `sparse_search`: 稀疏向量检索，适用于专有名词
  - `dense_search`: 密集向量检索，适用于语义检索
  - `hybrid_search`: 混合检索，结合稀疏和密集向量（推荐）
- 支持元数据筛选和条件查询
- 提供健康检查端点
- 支持 SSE (Server-Sent Events) 传输模式

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
MILVUS_COLLECTION_CRIMINAL_LAW=criminal_law
MILVUS_COLLECTION_CIVIL_CODE=civil_code
```

3. 启动服务：

```bash
# 启动 SSE 模式
python server.py --sse

# 启动标准模式
python server.py
```

## 混合检索支持

服务集成了 BGE-M3 模型，支持同时使用密集向量和稀疏向量进行混合检索，通过 RRF (Reciprocal Rank Fusion) 算法融合两种检索结果，提高检索质量。

### 混合检索工作流程

1. 使用 BGE-M3 模型将用户查询转换为密集向量和稀疏向量
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

1. criminal_law_query：使用过滤表达式进行精确查询
1. criminal_law_sparse_search：稀疏向量检索，适用于专有名词
1. criminal_law_dense_search：密集向量检索，适用于语义检索
1. criminal_law_hybrid_search：混合检索，适用于大多数情况

民法典检索：

1. civil_code_query：使用过滤表达式进行精确查询
1. civil_code_sparse_search：稀疏向量检索，适用于专有名词
1. civil_code_dense_search：密集向量检索，适用于语义检索
1. civil_code_hybrid_search：混合检索，适用于大多数情况
