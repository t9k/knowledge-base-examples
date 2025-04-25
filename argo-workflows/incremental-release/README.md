# 增量部署工作流

本工作流使用 Argo Workflows、Milvus 和嵌入模型实现知识库数据的增量部署流程。

## 概述

工作流执行以下步骤：

1. **同步文件**：从 S3 存储桶同步文件到工作空间，并识别新增、修改和删除的文件
2. **更新数据**：根据文件变更更新 Milvus 集合：
   - 实现两种原子操作：A，处理文件、分块、创建嵌入向量，并将数据插入到 Milvus 集合中；B，Milvus 集合中移除文件对应的数据
   - 创建文件调用原子操作 A，删除文件调用原子操作 B，修改文件调用原子操作 A 和 B

所有操作都记录到 `log.txt` 中，便于跟踪和调试。

## 文件说明

- `incremental-release-template.yaml`：Argo WorkflowTemplate 定义
- `configmap-workflow.yaml`：配置工作流的 ConfigMap
- `configmap-rclone-s3.yaml`：配置 S3 访问的 ConfigMap
- `pvc.yaml`：工作空间存储的 PersistentVolumeClaim
- `sync-files.sh`：从 S3 同步文件并识别变更的脚本
- `update-data.py`：处理文件变更并更新 Milvus 的 Python 脚本

## 配置说明

工作流需要以下参数：

- `collection-name`：Milvus 集合名称

ConfigMap incremental-release-config 包含以下环境变量：

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
   kubectl apply -f incremental-release-template.yaml
   ```

## 手动执行

手动运行工作流：

```bash
argo submit --from workflowtemplate/incremental-release \
  -p collection-name=<your-collection-name>
```
