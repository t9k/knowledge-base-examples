# 蓝绿部署工作流（法律）

本工作流使用 Argo Workflows、Milvus 和嵌入模型实现[法律数据](https://modelsphere.qy.t9kcloud.cn/datasets/t9k-ai/cn-laws)的蓝绿部署流程。

## 概述

工作流执行以下步骤：

1. **准备（prepare）**：检查参数格式，拉取后续步骤要执行的脚本
2. **下载文件（download）**：从 Model Sphere 下载法律数据文件
3. **插入数据（db_insert）**：创建新的 Milvus Database，以及其下的一个 Collection；处理法律数据文件：按 Markdown 标题和法条分段、提取元数据、创建嵌入向量，并将数据插入到 Database 下的 Collection 中

## 文件说明

- `bg-deploy-template.yaml`：Argo WorkflowTemplate 定义
- `configmap.yaml`：配置工作流的 ConfigMap
- `pvc.yaml`：工作空间存储的 PersistentVolumeClaim
- `download.sh`：从 Model Sphere 下载法律数据文件的脚本
- `db_insert.py`：创建 Milvus Database 及其下的 Collection，处理法律数据文件并插入数据到 Collection 的 Python 脚本

## 配置说明

工作流需要以下参数：

- `database-name`：Milvus Database 名称
- `collection-name`：Milvus Collection 名称
- `file-path`：法律数据文件路径

ConfigMap bg-deploy-law-config 包含以下环境变量：

- `MILVUS_URI`：Milvus 连接 URI
- `MILVUS_TOKEN`：Milvus 连接令牌
- `EMBEDDING_BASE_URL`：嵌入模型连接 URL
- `EMBEDDING_MODEL`：嵌入模型名称

## 数据处理

- 进行两级 chunking：使用 MarkdownHeaderTextSplitter 基于 Markdown 标题进行 chunking，再正则匹配法条，最终每个 chunk 是一个法条
- 使用 Enflame GCU 推理 bge-m3 模型生成稀疏嵌入向量，调用 Qwen3-Embedding-0.6B 模型生成密集嵌入向量
- 将提取的元数据存入到 Collection 中，供后续元数据过滤使用

## 蓝绿部署

该工作流实现了蓝绿部署模式，每次部署都会：

1. 创建新的 Milvus 数据库（通过 `database-name` 参数指定）
2. 在新数据库中创建集合（通过 `collection-name` 参数指定）
3. 将法律数据插入到新集合中

这样可以在不影响现有数据库的情况下进行部署，部署完成后可以将应用切换到新数据库。

## 镜像

步骤**插入数据（db-insert）**所使用的镜像由 [Dockerfile](./Dockerfile) 定义，其安装了 `pymilvus[model]`、`langchain_text_splitters` 等库，并预下载了 bge-m3 模型。

注意：镜像仓库 `registry.qy.t9kcloud.cn` 属于燧原庆阳集群，未开放公网访问。

## 运行

1. 创建 ConfigMap：
   ```bash
   kubectl apply -f configmap.yaml
   ```

2. 创建 PVC：
   ```bash
   kubectl apply -f pvc.yaml
   ```

3. 注册工作流模板：
   ```bash
   kubectl apply -f bg-deploy-template.yaml
   ```

4. 终端运行工作流：
  ```bash
  argo submit --from workflowtemplate/bg-deploy-laws \
    -p database-name=<your-database-name>
  ```

或进入 Argo Workflows 的 Web UI 操作以运行工作流。

## TODO

* 步骤 db-insert
  * 可以并行处理
* ConfigMap 中有一个 MILVUS_TOKEN 属于敏感信息
