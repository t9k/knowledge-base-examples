# Dify S3 同步工作流

这是一个 Argo 工作流，用于在 S3 存储桶和 Dify 知识库之间同步文档。它监控 S3 存储桶中的文档变更，并相应地更新 Dify 知识库中的文档。

## 功能特点

- 监控 S3 存储桶中的文档变更
- 支持多种文档格式（txt、pdf、md、doc、docx）
- 自动在 Dify 知识库中创建、更新或删除文档
- 可配置是否只处理最近修改的文件
- 可配置运行计划（默认：每小时）

## 配置说明

工作流接受以下参数：

- `host`：Dify 服务端点（必填）
- `api-key`：Dify API 密钥（必填）
- `dataset-id`：Dify 知识库的 ID（必填）
- `s3-bucket`：S3 存储桶名称（必填）
- `modified-only`：是否只处理修改时间发生变化的文件（默认："false"）

## 使用方法

1. 创建 PVC：
```bash
kubectl apply -f pvc.yaml
```

2. 应用 S3 配置（需要进行配置）：
```bash
kubectl apply -f s3-config.yaml
```

3. 应用工作流模板：
```bash
kubectl apply -f dify-s3-sync-template.yaml
```

4. 应用定时工作流（需要进行配置）：
```bash
kubectl apply -f dify-s3-sync-cron.yaml
```

或手动触发工作流：
```bash
argo submit -n argo --from cronworkflow/dify-s3-sync dify-s3-sync-template.yaml \
  --parameter host=<your-dify-host> \
  --parameter api-key=<your-api-key> \
  --parameter dataset-id=<your-dataset-id> \
  --parameter s3-bucket=<your-s3-bucket> \
  --parameter modified-only=false
```

## 工作流程

1. `pull-scripts`：从 GitHub 拉取步骤 2.3.4. 的脚本

2. `sync-s3-files`：从 S3 存储桶获取文件
    - 始终同步所有文件到 `/workspace/files/` 目录
    - 创建 `/workspace/s3_files.txt` 记录所有文件以及修改时间
    - 创建 `/workspace/modified_files.txt` 记录修改时间和上一次发生变化的文件

3. `fetch-dify-docs`：从 Dify 获取现有文档列表
    - 创建 `/workspace/dify_docs.json` 记录知识库的文档列表

4. `process-files`：处理文件并与 Dify 同步
    - 比照 `dify_docs.json` 和 `s3_files.txt`，删除 S3 中已不存在的文档
    - 比照 `dify_docs.json` 和 `s3_files.txt`（当 `modified-only` 为 `false`，或 `modified_files.txt`，当 `modified-only` 为 `true`）：
        - 为新文件创建新文档
        - 更新已修改文件的现有文档
