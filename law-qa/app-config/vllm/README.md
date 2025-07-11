# vLLM App 配置

本目录包含 vLLM App 配置文件，用于在平台上部署法律问答应用所需的 LLM、嵌入模型和重排模型。

## 配置文件

### qy-qwen3-32b.yaml
- **模型**: Qwen3-32B
- **用途**: 主要推理模型，用于生成回答和元数据提取
- **资源需求**: 4 CPU，64Gi 内存，2 GCU
- **API端点**: `/v1/chat/completions`

### qy-embedding.yaml  
- **模型**: Qwen3-Embedding-0.6B
- **用途**: 生成文本的密集向量表示
- **资源需求**: 4 CPU，64Gi 内存，1 GCU
- **API端点**: `/v1/embeddings`
- **输出**: 1024维向量

### qy-reranker.yaml
- **模型**: bge-reranker-v2-m3
- **用途**: 对检索结果进行重排序
- **资源需求**: 4 CPU，64Gi 内存，1 GCU
- **API端点**: `/rerank`
