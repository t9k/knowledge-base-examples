# vLLM

当前目录包含 vLLM 应用的配置文件，用于在平台上部署法律问答应用所需的 LLM、嵌入模型和重排模型。

## 配置文件

注意：

1. 需要使用以下所有配置文件分别安装 vLLM 应用。
2. 如使用 Enflame GCU 加速推理，换用名称带有 `-gcu` 后缀的配置文件。

### qwen3-32b.yaml

- **模型**: Qwen3-32B
- **用途**: 主要推理模型，用于生成回答
- **资源需求**: 4 CPU，64Gi 内存，2 GPU
- **API端点**: `/v1/chat/completions`

### embedding.yaml  

- **模型**: Qwen3-Embedding-0.6B
- **用途**: 生成文本的密集向量表示
- **资源需求**: 4 CPU，64Gi 内存，1 GPU
- **API端点**: `/v1/embeddings`
- **输出**: 1,024 维向量

### reranker.yaml

- **模型**: bge-reranker-v2-m3
- **用途**: 对检索结果进行重排序
- **资源需求**: 4 CPU，64Gi 内存，1 GPU
- **API端点**: `/rerank`
