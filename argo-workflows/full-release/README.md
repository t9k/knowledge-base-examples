# 全量部署工作流

本工作流使用 Argo Workflows、Milvus 和嵌入模型实现知识库数据的全量部署流程。

## 概述

工作流执行以下步骤：

1. **同步文件**：从 S3 存储桶同步文件到工作空间
2. **插入数据**：创建 Milvus 集合；处理文件、分块、创建嵌入向量，并将数据插入到 Milvus 集合中

## 文件说明

- `full-release-template.yaml`：Argo WorkflowTemplate 定义
- `configmap-workflow.yaml`：配置工作流的 ConfigMap
- `configmap-rclone-s3.yaml`：配置 S3 访问的 ConfigMap
- `pvc.yaml`：工作空间存储的 PersistentVolumeClaim
- `sync-files.sh`：从 S3 同步文件的脚本
- `insert-data.py`：处理文件并插入到 Milvus 的 Python 脚本

## 配置说明

工作流需要以下参数：

- `collection-name`：Milvus 集合名称

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
   kubectl apply -f configmap-workflow.yaml
   kubectl apply -f configmap-rclone-s3.yaml
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
