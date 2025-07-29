# 外部知识库 API 服务

这是一个基于 Flask 和 Milvus 的外部知识库 API 服务实现，支持对接 Dify 的外部知识库接口，提供标准向量检索和混合检索功能。

## 功能

- 实现了 [Dify 外部知识库 API](https://docs.dify.ai/zh-hans/guides/knowledge-base/api-documentation/external-knowledge-api-documentation)
- Qwen3-Embedding-4B 模型提供密集嵌入，BGE-M3 模型提供稀疏嵌入，支持混合检索
- 集成 Milvus 向量数据库，支持三种检索模式：
  - `dense`: 仅使用密集向量检索
  - `sparse`: 仅使用稀疏向量检索
  - `hybrid`: 同时使用密集和稀疏向量进行混合检索（默认）
- 支持元数据筛选
- 详细的日志和错误信息

## Kubernetes 部署

修改 `k8s.yaml` 中的 ConfigMap 和 Secret 的配置，然后执行以下命令以部署：

```bash
kubectl apply -f k8s.yaml
```

如要部署多个服务，手动替换 dify-ekb-server 为其增加名称后缀（镜像 ID 除外），如 dify-ekb-server-law、dify-ekb-server-criminal-case。

## 混合检索支持

服务调用 Qwen3-Embedding-0.6B 模型提供密集嵌入，集成 BGE-M3 模型提供稀疏嵌入，支持同时使用密集向量和稀疏向量进行混合检索，通过 RRF (Reciprocal Rank Fusion) 算法融合两种检索结果，提高检索质量。

### 混合检索工作流程

1. 使用 Qwen3-Embedding-0.6B 和 BGE-M3 模型将用户查询转换为密集向量和稀疏向量
2. 分别用密集向量和稀疏向量在 Milvus 中进行检索
3. 使用 RRF 算法对两种检索结果进行融合排序
4. 返回包含元数据的检索结果

## Milvus Collection 要求

为了使服务正常工作，scenario 为 `law` 时，Milvus Collection 需要包含以下字段：

* `chunk_id` (VarChar)：chunk 的唯一标识符
* `chunk` (VarChar)：chunk 的内容
* `law` (VarChar)：所属法律
* `part` (VarChar)：所属编
* `chapter` (VarChar)：所属章
* `section` (VarChar)：所属节
* `article` (Int64)：法条序号（0 表示没有序号）
* `article_amended` (Int64)：修正的刑法法条序号（0 表示没有修正刑法法条）
* `sparse_vector` (SparseFloatVector)：稀疏嵌入
* `dense_vector` (FloatVector)：密集嵌入

scenario 为 `criminal_case` 时，Milvus Collection 需要包含以下字段：

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

scenario 为 `civil_case` 时，Milvus Collection 需要包含以下字段：

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

## API 使用示例

### 基本请求

```bash
curl -X POST http://localhost:5001/retrieval \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer t9k-12345" \
  -d '{
    "knowledge_id": "your_milvus_collection",
    "query": "你的问题",
    "retrieval_setting": {
      "top_k": 2,
      "score_threshold": 0.5
    }
  }'
```

### 带检索模式的请求

```bash
curl -X POST http://localhost:5001/retrieval \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer t9k-12345" \
  -d '{
    "knowledge_id": "your_milvus_collection",
    "query": "你的问题",
    "search_mode": "dense",
    "retrieval_setting": {
      "top_k": 2,
      "score_threshold": 0.5
    }
  }'
```

### 带元数据条件的请求

```bash
curl -X POST http://localhost:5001/retrieval \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer t9k-12345" \
  -d '{
    "knowledge_id": "your_milvus_collection",
    "query": "你的问题",
    "retrieval_setting": {
      "top_k": 2,
      "score_threshold": 0.5
    },
    "metadata_condition": {
      "logical_operator": "and",
      "conditions": [
        {
          "name": ["category"],
          "comparison_operator": "contains",
          "value": "AI"
        },
        {
          "name": ["date"],
          "comparison_operator": "after",
          "value": "2023-01-01"
        }
      ]
    }
  }'
```

### 示例响应

```json
{
  "records": [
    {
      "metadata": {
        "relevant_articles": "...",
        "accusation": "...",
        "punish_of_money": "...",
        "criminals": "...",
        "imprisonment": "..."
      },
      "score": 0.98,
      "title": "doc_id_1",
      "content": "这是外部知识的文档内容。"
    },
    {
      "metadata": {
        "relevant_articles": "...",
        "accusation": "...",
        "punish_of_money": "...",
        "criminals": "...",
        "imprisonment": "..."
      },
      "score": 0.66,
      "title": "doc_id_2",
      "content": "这是另一个相关文档。"
    }
  ]
}
```

## 错误处理

服务会返回适当的错误代码和消息，例如：

| 错误代码 | 描述                        |
| -------- | --------------------------- |
| 1001     | 无效的 Authorization 头格式 |
| 1002     | 授权失败                    |
| 2001     | 知识库不存在或请求参数错误  |
| 3001     | Milvus 连接错误             |

错误响应示例:

```json
{
  "error_code": 2001,
  "error_msg": "Knowledge base 'your_collection' not found"
}
```
