# Dify S3 同步工作流 (蓝绿部署版)

这是一个 Argo 工作流，用于在 S3 存储桶和 Dify 知识库之间同步文档。它监控 S3 存储桶中的文档变更，并相应地更新 Dify 知识库中的文档。该版本实现了蓝绿部署策略，允许零停机时间的升级和快速回滚能力。

**与[基础版本](../basic/README.md)区别**：此版本维护两个独立的环境（蓝和绿），在它们之间交替升级，而基础版本只有单一环境。

## 功能特点

- 监控 S3 存储桶中的文档变更
- 支持多种文档格式（txt、pdf、md、doc、docx）
- 自动在 Dify 知识库中创建、更新或删除文档
- 可配置是否处理全部文件或仅处理新建和修改的文件
- 可配置运行计划（默认：每天下午3点）
- 蓝绿部署模式，维护两个独立环境并交替升级，实现零停机升级和快速回滚

## 配置说明

工作流接受以下参数：

- `host`：Dify 服务端点（必填）
- `api-key`：Dify API 密钥（必填）
- `dataset-id-blue`：蓝色环境使用的 Dify 知识库 ID（必填）
- `dataset-id-green`：绿色环境使用的 Dify 知识库 ID（必填）
- `s3-bucket`：S3 存储桶名称（必填）
- `alway-push-all-files`：是否处理所有文件而不只是新建和修改的文件（默认："true"）

## 目录结构

工作流维护以下目录结构：

```
/workspace/
├── files/                # 共享文件目录（从 S3 同步）
│   ├── mi.txt
│   └── news1.txt
├── scripts/              # 共享脚本目录
│   ├── fetch_dify_docs.sh
│   ├── process_files.sh
│   └── sync_s3_files.sh
├── blue/                 # 蓝色环境
│   ├── dify_docs.json
│   ├── files_to_create.txt
│   ├── files_to_modify.txt
│   ├── files_to_delete.txt
│   └── s3_files.txt
├── green/                # 绿色环境
│   ├── dify_docs.json
│   ├── files_to_create.txt
│   ├── files_to_modify.txt
│   ├── files_to_delete.txt
│   └── s3_files.txt
└── flag.txt              # 记录当前活动环境（blue 或 green）
```

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
  --parameter dataset-id-blue=<your-blue-dataset-id> \
  --parameter dataset-id-green=<your-green-dataset-id> \
  --parameter s3-bucket=<your-s3-bucket> \
  --parameter alway-push-all-files=true
```

## 工作流程

1. `prepare`：准备工作
    - 从 GitHub 拉取后续步骤的脚本并存储到共享脚本目录
    - 检查 `/workspace/flag.txt` 的内容（上一次的活动环境）：
        - 若为 `blue`，则修改为 `green`，确定绿色为当前活动环境
        - 若不为 `blue`，或文件不存在，则修改为 `blue`，确定蓝色为当前活动环境

2. 并行步骤：

    - `fetch-dify-docs`：获取 Dify 知识库文档列表
        - 选择当前活动环境对应的数据集 ID
        - 创建活动环境的 `dify_docs.json` 记录文档列表

    - `sync-s3-files`：从 S3 存储桶获取文件
        - 始终同步所有文件到共享的 `/workspace/files/` 目录
        - 创建活动环境的 `s3_files.txt` 记录文件列表，包含文件名和修改时间
        - 创建活动环境的 `files_to_create.txt` 记录新建的文件
        - 创建活动环境的 `files_to_modify.txt` 记录修改的文件
        - 创建活动环境的 `files_to_delete.txt` 记录删除的文件

3. `update-dify-knowledge-base`：更新当前活动环境对应的 Dify 知识库
   
    - 对于每个要删除的文件 (`files_to_delete.txt`)，如果文件存在于文档列表中，则删除该文档

    - 根据 `alway-push-all-files` 参数决定上传的文件：
        - 若为 `true`，上传文件列表 (`s3_files.txt`) 中的所有文件
        - 若为 `false`，仅上传要新建和修改的文件 (`files_to_create.txt` 和 `files_to_modify.txt`)
        - 对每个上传的文件：
            - 如果文件存在于文档列表中，则更新该文档
            - 如果文件不存在于文档列表中，则创建文档

## 回滚说明

如果版本升级后出现问题（假设最近一次升级到绿色环境）：

1. 将应用路由切换到蓝色环境，将 `/workspace/flag.txt` 的值改为 `blue`；
2. 检查、修复绿色环境的问题；
3. 将应用路由切换到绿色环境，将 `/workspace/flag.txt` 的值改为 `green`。
