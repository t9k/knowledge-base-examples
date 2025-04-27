# 全量发布工作流

本工作流使用 Argo Workflows、Milvus 和嵌入模型实现知识库数据的全量发布流程。

## 概述

工作流执行以下步骤：

1. **同步文件**：从 S3 存储桶同步文本文件到工作空间
2. **插入数据**：创建新的 Milvus Collection；处理文本文件：分块、创建嵌入向量，并将数据插入到该 Collection 中

## 文件说明

- `full-release-template.yaml`：Argo WorkflowTemplate 定义
- `configmap.yaml`：配置工作流的 ConfigMap
- `secret.yaml`：配置 S3 访问的 Secret
- `pvc.yaml`：工作空间存储的 PersistentVolumeClaim
- `sync-files.sh`：从 S3 同步文件的脚本
- `insert-data.py`：创建 Milvus Collection，处理所有文件并插入数据到 Collection 的 Python 脚本

## 配置说明

工作流需要以下参数：

- `database-name`：Milvus Database 名称
- `collection-name`：Milvus Collection 名称

ConfigMap full-release-config 包含以下环境变量：

- `S3_PATH`：S3 文件匹配路径
- `MILVUS_URI`：Milvus 连接 URI
- `EMBEDDING_BASE_URL`：嵌入模型 API 的基础 URL
- `EMBEDDING_MODEL`：嵌入模型名称
- `EMBEDDING_DIM`：嵌入向量维度

ConfigMap rclone-config 包含 rclone 配置文件，用于访问 S3 存储桶。

## 设置步骤

1. 创建 ConfigMap：
   ```bash
   kubectl apply -f configmap.yaml
   kubectl apply -f secret.yaml
   ```

2. 创建 PVC：
   ```bash
   kubectl apply -f pvc.yaml
   ```

3. 注册工作流模板：
   ```bash
   kubectl apply -f full-release-template.yaml
   ```

## 手动执行

手动运行工作流：

```bash
argo submit --from workflowtemplate/full-release \
  -p collection-name=<your-collection-name>
```

## TODO

* ConfigMap 中有一个 MILVUS_TOKEN 属于敏感信息
* 内存优化
