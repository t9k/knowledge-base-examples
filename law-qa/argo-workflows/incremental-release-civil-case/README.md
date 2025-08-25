# 增量发布工作流（民事案件）

本工作流使用 Argo Workflows、Milvus 和嵌入模型实现[民事案件数据](https://modelsphere.nc201.t9kcloud.cn/datasets/xyx/judgment-documents)的增量发布流程。

## 概述

工作流执行以下步骤：

1. **准备（prepare）**：检查参数格式，拉取后续步骤要执行的脚本
2. **下载文件（download）**：从 Model Sphere 下载民事案例数据文件
3. **插入数据（db_insert）**：创建或更新指定名称的 2 个 Milvus Collection，处理案例数据文件：
   1. 若以来源（文件名）查询 entities 不存在，则分块、创建嵌入向量，并插入数据
   2. 否则：
      1. 若以最后修改时间查询 entities 不存在，则先删除有相同来源（文件名）的数据，再分块、创建嵌入向量，并插入数据
      2. 否则，直接返回

## 文件说明

- `incremental-release-template.yaml`：Argo WorkflowTemplate 定义
- `configmap.yaml`：配置工作流的 ConfigMap
- `pvc.yaml`：工作空间存储的 PersistentVolumeClaim
- `download.sh`：从 Model Sphere 下载民事案例数据文件的脚本
- `db_insert.py`：创建或更新 Milvus Collection，处理案例数据文件并插入数据到 Collection 的 Python 脚本

## 配置说明

工作流需要以下参数：

- `database-name`：Milvus Database 名称

ConfigMap incremental-release-civil-case-config 包含以下环境变量：

- `MILVUS_URI`：Milvus 连接 URI
- `MILVUS_TOKEN`：Milvus 连接令牌
- `PARENT_COLLECTION_NAME`：父集合名称（固定名称，创建在指定 Database 中）
- `PARENT_CHUNK_SIZE`：父集合分块大小
- `PARENT_CHUNK_OVERLAP`：父集合分块重叠大小
- `COLLECTION_NAME`：集合名称（固定名称，创建在指定 Database 中）
- `CHUNK_SIZE`：子集合分块大小
- `CHUNK_OVERLAP`：子集合分块重叠大小
- `EMBEDDING_BASE_URL`：嵌入模型连接 URL
- `EMBEDDING_MODEL`：嵌入模型名称
- `CHAT_BASE_URL`：聊天模型连接 URL
- `CHAT_MODEL`：聊天模型名称
- `IS_PARENT_CHILD`：是否启用父子分段
- `IS_LLM_EXTRACT`：是否启用 LLM 提取元数据
- `LLM_WORKERS`：LLM 线程数

## 数据处理

- 使用 RecursiveCharacterTextSplitter 进行父子分段，保持结构完整性的同时避免单个 chunk 过长
- 使用 Enflame GCU 推理 bge-m3 模型生成稀疏嵌入向量，调用 Qwen3-Embedding-0.6B 模型生成密集嵌入向量
- 调用 Qwen3-32B 模型提取元数据，与已有的元数据一并存入到 Collection 中，供后续元数据过滤使用

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
   kubectl apply -f incremental-release-template.yaml
   ```

4. 终端运行工作流：
  ```bash
  argo submit --from workflowtemplate/incremental-release-civil-cases \
    -p database-name=<your-database-name>
  ```

或进入 Argo Workflows 的 Web UI 操作以运行工作流。

## TODO

* 步骤 db-insert
  * 可以并行处理
* ConfigMap 中有一个 MILVUS_TOKEN 属于敏感信息
