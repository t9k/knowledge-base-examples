# 蓝绿部署工作流

本工作流使用 Argo Workflows、Milvus 和嵌入模型实现知识库数据的蓝绿部署流程。

## 概述

工作流执行以下步骤：

1. **同步文件**：从多个 S3 存储桶路径同步文本和图像文件到工作空间的不同子目录
2. **发布版本**：
  1. **创建数据库**：创建新的 Milvus Database，以及其下的多个 Collection
  2. **插入文本数据**：处理文本文件：分块、创建嵌入向量，并将数据插入到 Database 的一个 Collection 中
  3. **插入图像数据**：处理图像文件：中心裁剪，归一化，创建嵌入向量，并将数据插入到 Database 的另一个 Collection 中

## 文件说明

- `bg-deploy-template.yaml`：Argo WorkflowTemplate 定义
- `configmap.yaml`：配置工作流的 ConfigMap
- `secret.yaml`：配置 S3 访问的 Secret
- `pvc.yaml`：工作空间存储的 PersistentVolumeClaim
- `sync-files.sh`：从 S3 同步文件的脚本
- `publish-release.py`：创建 Milvus Database 及其下的 Collection，处理文本/图像文件并插入数据到多个 Collection 的 Python 脚本

## 配置说明

工作流需要以下参数：

- `database-name`：Milvus Database 名称

ConfigMap bg-deploy-config 包含以下环境变量：

- `S3_PATH_TEXT`：文本文件的 S3 路径
- `S3_PATH_IMAGE`：图像文件的 S3 路径
- `COLLECTION_NAME_TEXT`：文本数据的集合名称（固定名称，创建在指定 Database 中）
- `COLLECTION_NAME_IMAGE`：图像数据的集合名称（固定名称，创建在指定 Database 中）
- `MILVUS_URI`：Milvus 连接 URI
- `MILVUS_TOKEN`：Milvus 连接令牌
- `TEXT_EMBEDDING_BASE_URL`：文本嵌入模型 API 的基础 URL
- `TEXT_EMBEDDING_MODEL`：文本嵌入模型名称
- `TEXT_EMBEDDING_DIM`：文本嵌入向量维度
- `IMAGE_EMBEDDING_MODEL`：图像嵌入模型名称
- `IMAGE_EMBEDDING_DIM`：图像嵌入向量维度

ConfigMap rclone-config 包含 rclone 配置文件，用于访问 S3 存储桶。

## 文本和图像处理

工作流支持处理文本和图像数据：

### 文本处理
- 支持的文件格式：`.txt`、`.md`
- 使用在线API（通过环境变量 `TEXT_EMBEDDING_BASE_URL` 指定）生成嵌入向量
- 文本会按段落分块，保持语义完整性

### 图像处理
- 支持的文件格式：`.jpg`、`.jpeg`、`.png`
- 使用指定的预训练图像编码模型生成嵌入向量
- 图像处理流程：
  1. 加载图像并调整大小到 384×384 像素
  2. 应用标准化处理
  3. 批量生成图像嵌入向量
  4. 向量归一化后存入 Milvus

## 蓝绿部署

该工作流实现了蓝绿部署模式，每次部署都会：

1. 创建新的 Milvus 数据库（通过 `database-name` 参数指定）
2. 在新数据库中创建固定名称的集合（`COLLECTION_NAME_TEXT` 和 `COLLECTION_NAME_IMAGE`）
3. 将最新的文本和图像数据插入到新集合中

这样可以在不影响现有数据库的情况下进行部署，部署完成后可以将应用切换到新数据库。

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
   kubectl apply -f bg-deploy-template.yaml
   ```

## 手动执行

手动运行工作流：

```bash
argo submit --from workflowtemplate/bg-deploy \
  -p database-name=<your-database-name>
```

## TODO

* 步骤 publish-release
  * 文本和图像处理分开，可以并行处理
  * 离线的图案处理请求一个 GPU
* 用户可以指定任意个文本数据源和图像数据源，以及相应的 Collection 名称
* ConfigMap 中有一个 MILVUS_TOKEN 属于敏感信息
