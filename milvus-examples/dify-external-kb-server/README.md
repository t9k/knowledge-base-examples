# 外部知识库 API 服务

这是一个基于 Flask 和 Milvus 的外部知识库 API 服务实现，支持对接 Dify 的外部知识库接口，提供标准向量检索和混合检索功能。

## 功能

- 实现了 [Dify 外部知识库 API](https://docs.dify.ai/zh-hans/guides/knowledge-base/api-documentation/external-knowledge-api-documentation)
- 支持 bge-m3 模型的混合检索 (dense + sparse)，提高检索质量
- 集成 Milvus 向量数据库，支持三种检索模式：
  - `dense`: 仅使用密集向量检索
  - `sparse`: 仅使用稀疏向量检索
  - `hybrid`: 同时使用密集和稀疏向量进行混合检索（默认）
- 支持元数据筛选
- 详细的日志和错误信息

## 部署

### Docker 部署

构建并运行容器：

```bash
docker run -p 5001:5001 -e "MILVUS_HOST=host.docker.internal" milvus-external-knowledge
```

## 混合检索支持

服务集成了 bge-m3 模型，支持同时使用密集向量和稀疏向量进行混合检索，通过 RRF (Reciprocal Rank Fusion) 算法融合两种检索结果，提高检索质量。

### 混合检索工作流程

1. 使用 bge-m3 模型将用户查询转换为密集向量和稀疏向量
2. 分别用密集向量和稀疏向量在 Milvus 中进行检索
3. 使用 RRF 算法对两种检索结果进行融合排序
4. 对结果进行后处理并返回给客户端

## Milvus Collection 要求

为了使服务正常工作，Milvus Collection 需要包含以下字段：

### 必需字段

- `dense_vector`: 密集向量字段
- `sparse_vector`: 稀疏向量字段
- `fact`: 文档内容
- `pk`: 文档ID/标题

### 可选元数据字段

- `relevant_articles`
- `accusation`
- `punish_of_money`
- `criminals`
- `imprisonment`
- `life_imprisonment`
- `death_penalty`

> 注意：字段名可在代码中的 `output_fields` 列表调整以匹配您的集合结构。

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

## 环境变量

服务支持以下环境变量：

- `DEBUG_MODE`: 设置为 "true" 启用详细日志记录，默认为 "false"

例如：

```bash
DEBUG_MODE=true python app.py
```
