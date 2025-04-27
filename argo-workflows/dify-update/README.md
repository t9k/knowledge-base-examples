# Dify S3 同步工作流

这是一个 Argo 工作流，用于在 S3 存储桶和 Dify 知识库之间同步文档。它监控 S3 存储桶中的文档变更，并相应地更新 Dify 知识库中的文档。

## 功能特点

- 监控 S3 存储桶中的文档变更
- 支持多种文档格式（txt、pdf、md、doc、docx）
- 自动在 Dify 知识库中创建、更新或删除文档
- 可配置是否处理全部文件或仅处理新建和修改的文件
- 可配置运行计划（默认：每小时）

## 配置说明

工作流接受以下参数：

- `host`：Dify 服务端点（必填）
- `api-key`：Dify API 密钥（必填）
- `dataset-id`：Dify 知识库的 ID（必填）
- `s3-bucket`：S3 存储桶名称（必填）
- `alway-push-all-files`：是否处理所有文件而不只是新建和修改的文件（默认："true"）

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
  --parameter alway-push-all-files=true
```

## 工作流程

1. `pull-scripts`：从 GitHub 拉取后续步骤的脚本并存储到共享脚本目录

2. 并行步骤：

    - `fetch-dify-docs`：获取 Dify 知识库文档列表
        - 创建 `/workspace/dify_docs.json` 记录文档列表

    - `sync-s3-files`：从 S3 存储桶同步文件
        - 始终同步所有文件到 `/workspace/files/` 目录
        - 创建 `/workspace/s3_files.txt` 记录文件列表，包含文件名和修改时间
        - 创建 `/workspace/files_to_create.txt` 记录新建的文件
        - 创建 `/workspace/files_to_modify.txt` 记录修改的文件
        - 创建 `/workspace/files_to_delete.txt` 记录删除的文件

3. `update-dify-knowledge-base`：更新 Dify 知识库

    - 对于每个要删除的文件 (`files_to_delete.txt`)，如果文件存在于文档列表中，则删除该文档

    - 根据 `alway-push-all-files` 参数决定上传的文件：
        - 若为 `true`，上传文件列表 (`s3_files.txt`) 中的所有文件
        - 若为 `false`，仅上传要新建和修改的文件 (`files_to_create.txt` 和 `files_to_modify.txt`)
        - 对每个上传的文件：
            - 如果文件存在于文档列表中，则更新该文档
            - 如果文件不存在于文档列表中，则创建文档
