# 智能法律问答应用

这是一个智能法律问答应用的项目，旨在构建一个能够准确、简明回答用户法律问题的智能助手。

## 项目概述

该项目构建了一个完整的法律知识检索和问答系统，包含：

- **法律数据处理**：处理刑事案例、民事案例和法律法规数据
- **向量检索服务**：基于 MCP 协议的多种检索服务
- **模型部署**：支持多种 LLM 和嵌入模型的部署配置
- **Agent 实现**：使用 Qwen Agent 实现智能法律问答 Agent

## 项目结构

```
law-qa/
├── agent/               # Agent 实现
├── app-configs/         # 应用部署配置
│   ├── milvus/          # Milvus 向量数据库配置
│   └── vllm/            # vLLM 模型服务配置
├── argo-workflows/      # Argo Workflows 工作流
│   ├── bg-deploy-civil-case/    # 蓝绿发布民事案件数据
│   ├── bg-deploy-criminal-case/ # 蓝绿发布刑事案件数据
│   └── bg-deploy-law/           # 蓝绿发布法律法规数据
├── data/                # 数据处理脚本和文档
└── mcp-servers/         # MCP 检索服务
    ├── case-searcher/   # 案例检索服务
    ├── law-searcher/    # 法条检索服务
    └── reranker/        # 重排服务
```

## 部署流程

### 安装平台应用

**目标**

在平台上安装/部署法律问答应用所依赖的各种应用程序。

**步骤**

1. 创建一个名为 law-qa、大小 100Gi 的 PVC，然后安装一个同名的 JupyterLab 应用挂载该 PVC；进入 JupyterLab 应用的 UI，下载模型文件：

```bash
modelscope download --model "Qwen/Qwen3-32B" --local_dir "./Qwen3-32B"
modelscope download --model "Qwen/Qwen3-Embedding-0.6B" --local_dir "./Qwen3-Embedding-0.6B"
modelscope download --model "Qwen/Qwen3-Reranker-4B" --local_dir "./Qwen3-Reranker-4B"
```

2. 安装多个 [vLLM 应用](./platform-apps/vllm/README.md)；
3. 安装 [Milvus 应用](./platform-apps/milvus/README.md)；
4. 安装 [Open WebUI 应用](./platform-apps/open-webui/README.md)；
5. 安装 Argo Workflows 应用。

说明：

- 对于步骤 2、3、4，点击链接查看相应的 README，按照 README 使用配置文件安装应用。
- 对于步骤 5，使用默认配置安装应用即可。

**验证方法**

上述应用均已就绪。

### 导入数据

**目标**

从托管服务下载数据文件，处理并导入到 Milvus 向量数据库。

**步骤**

1. 回到 JupyterLab 应用的 UI，拉取当前项目：

```bash
git clone https://github.com/t9k/knowledge-base-examples.git
```

2. 执行脚本并提供必要的参数，以启动处理和导入数据的工作流：

    ```bash
    ./knowledge-base-examples/law-qa/argo-workflows/run_bg_workflows.sh \
      --milvus-uri <MILVUS_URI> \
      --embedding-base-url <EMBEDDING_BASE_URL> \
      --chat-base-url <CHAT_BASE_URL> \
      --git-repo-civil-case https://www.modelscope.cn/datasets/qazwsxplkj/cn-judgment-docs-demo.git \
      --git-repo-criminal-case https://www.modelscope.cn/datasets/qazwsxplkj/CAIL2018.git \
      --git-repo-cn-law https://www.modelscope.cn/datasets/qazwsxplkj/cn-laws.git \
      --is-llm-extract=true \
      --llm-workers=4 \
      TODO: --gpu-resource=nvidia.com/gpu
    ```

    这一步骤启动了处理和导入民事案件、刑事案件、法律法规 3 个 Argo Workflow。

说明：

- 上述过程启用了调用 LLM 提取元数据，这将成为数据处理与导入的瓶颈，导致工作流的运行时间长达数周。可采取以下优化措施：
  1. 安装 [vLLM 应用](./platform-apps/vllm/README.md)部署 Qwen3-32B 模型时，增加副本数量，并增大 `--llm-workers` 参数的值（具体数值需要测试，以充分利用所有 vLLM 实例的并行处理能力，同时避免 OOM），提升并行度。
  2. fork 数据集仓库，删除部分文件以缩小数据规模。
  3. 禁用 LLM 提取元数据，设置 `--is-llm-extract=false`。
- 上述过程需要 NVIDIA GPU 加速（稀疏）嵌入模型的离线推理。如换用 Enflame GPU，请将 `--gpu-vendor=nvidia` 替换为 `--gpu-vendor=enflame`。
- [此文档](./argo-workflows/data.md)记录了数据的基本信息，以及处理与入库的基本流程。

**验证方法**

<!-- 进入 Argo Workflows 应用的 UI，查看到 3 个工作流均已完成。 -->

持续监控 3 个工作流的状态，直到全部变为 Succeeded。通过执行以下 bash 脚本：

```bash
# 监控 3 个工作流的状态
./knowledge-base-examples/law-qa/argo-workflows/check_bg_workflows.sh
```

打印结果应类似于：

```
Monitoring prefixes: bg-deploy-civil-case- bg-deploy-criminal-case- bg-deploy-law-

[2025-09-11 12:34:56] Current status
PREFIX                               WORKFLOW_NAME                    PHASE       
------------------------------------ -------------------------------- ------------
bg-deploy-civil-case-                bg-deploy-civil-case-m8hnh       Running     
bg-deploy-criminal-case-             bg-deploy-criminal-case-8llzc    Running     
bg-deploy-law-                       bg-deploy-law-rjrgl              Running

...

[2025-09-12 12:34:56] Current status
PREFIX                               WORKFLOW_NAME                    PHASE       
------------------------------------ -------------------------------- ------------
bg-deploy-civil-case-                bg-deploy-civil-case-m8hnh       Succeeded     
bg-deploy-criminal-case-             bg-deploy-criminal-case-8llzc    Succeeded     
bg-deploy-law-                       bg-deploy-law-rjrgl              Succeeded
```

### 启动 MCP 服务

**目标**

提供 MCP 工具和资源。

**步骤**

1. 回到 JupyterLab 应用的 UI，执行脚本并提供必要的参数，以启动 MCP 服务：

    ```bash
    ./knowledge-base-examples/law-qa/mcp-servers/run_mcp_servers.sh \
      --milvus-uri <MILVUS_URI> \
      --milvus-db default \
      --embedding-base-url <EMBEDDING_BASE_URL> \
      --reranker-base-url <RERANKER_BASE_URL> \
      --enable-auth false \
      --gpu-vendor nvidia
    ```

    这一步骤启动了案件检索、法条检索和重排序 3 个 MCP 服务。对于每个服务，创建了以下 Kubernetes 资源：

    - Deployment：部署服务
    - Service：提供服务的访问入口
    - VirtualService：Istio 流量路由规则
    - ConfigMap：存储相应的配置

说明：

- 3 个 MCP 服务的端点分别为：
  - `http://mcp-server-case-searcher-service.<YOUR_NAMESPACE>.svc.cluster.local/mcp/case-searcher/mcp/` 或 `http://mcp-server-case-searcher-service/mcp/case-searcher/mcp/`
  - `http://mcp-server-law-searcher-service.<YOUR_NAMESPACE>.svc.cluster.local/mcp/law-searcher/mcp/` 或 `http://mcp-server-law-searcher-service/mcp/law-searcher/mcp/`
  - `http://mcp-server-reranker-service.<YOUR_NAMESPACE>.svc.cluster.local/mcp/reranker/mcp/` 或 `http://mcp-server-reranker-service/mcp/reranker/mcp/`
- 在 qy 集群部署的 MCP 服务暴露了**公网访问**，URL 如下：

| 服务         | URL                                                |
| ------------ | -------------------------------------------------- |
| 案件检索服务 | https://home.qy.t9kcloud.cn/mcp/case-searcher/mcp/ |
| 法条检索服务 | https://home.qy.t9kcloud.cn/mcp/law-searcher/mcp/  |
| 重排序服务   | https://home.qy.t9kcloud.cn/mcp/reranker/mcp/      |

**验证方法**

从集群内部或公网向 3 个 MCP 服务发起请求，都得到正确的响应。通过执行以下 python 脚本：

```bash
pip install fastmcp
# 列举每个 MCP 服务器的工具、资源和提示词
python ./knowledge-base-examples/law-qa/mcp-servers/test_mcp_servers.py <CASE_SEARCHER_MCP_SERVER_URL> <LAW_SEARCHER_MCP_SERVER_URL> <RERANKER_MCP_SERVER_URL>
```

打印结果应类似于：

```
===== https://home.qy.t9kcloud.cn/mcp/case-searcher/mcp/ =====
Tools:
[Tool(name='criminal_case_query', description='\n    使用过滤表达式查询刑事案例 Collection。\n\n    Collection 字段（元数据）说明：...
Resources:
[]
Prompts:
[]

===== https://home.qy.t9kcloud.cn/mcp/law-searcher/mcp/ =====
Tools:
[Tool(name='law_query', description='\n    使用过滤表达式查询法律 Collection。\n\n    Collection 字段（元数据）说明：...
Resources:
[Resource(uri=Url('data://criminal-law-contents'), name='criminal_law_contents', description='刑法的目录。作为构建与编、章、节相关的过滤表达式时的参考。', ...
Prompts:
[]

===== https://home.qy.t9kcloud.cn/mcp/reranker/mcp/ =====
Tools:
[Tool(name='rerank', description='根据查询文本对文档列表进行重排序。适用于检索结果较多、较复杂的情况。...
Resources:
[]
Prompts:
[]
```

### 启动 Agent 服务

**目标**

提供兼容 OpenAI API 的 Agent 服务。

**步骤**

1. 回到 JupyterLab 应用的 UI，执行脚本并提供必要的参数，以启动 MCP 服务：

    ```bash
    ./knowledge-base-examples/law-qa/agent/run_agent_server.sh \
      --chat-base-url <CHAT_BASE_URL> \
      --law-searcher-url <LAW_SEARCHER_MCP_SERVER_URL> \
      --case-searcher-url <CASE_SEARCHER_MCP_SERVER_URL> \
      --reranker-url <RERANKER_MCP_SERVER_URL>
    ```

    这一步骤创建了以下 Kubernetes 资源：

    - Deployment：部署 Agent 服务
    - Service：提供 Agent 服务的访问入口
    - Secret：存储 LLM 推理服务的 API Key

说明：

- Agent 的服务端点为 `http://law-agent.<YOUR_NAMESPACE>.svc.cluster.local:8001/v1` 或 `http://law-agent:8001/v1`

**验证方法**

从集群内部向 Agent 服务发起请求，得到正确的响应。通过执行以下命令：

```bash
curl -X POST http://law-agent:8001/v1/chat/completions \
  -H "Authorization: Bearer dummy" \
  -H "Content-Type: application/json" \
  -d '{"model":"dummy","messages":[{"role":"user","content":"你好"}],"temperature":0.0}'
```

打印结果应类似于：

```json
{"id":"chatcmpl-1757495878","object":"chat.completion","created":1757495878,"model":"dummy","choices":[{"index":0,"message":{"role":"assistant","content":"你好！我是法小助，专业的法律智能助手。请问你今天需要了解什么法律问题？我会尽力提供准确、简明的解答。","reasoning_content":"好的，用户发来“你好”，这是一个简单的问候。首先需要按照意图识别的框架来分析。\n\n1. **意图识别**：用户只是打招呼，没有提出具体问题。可能属于其他类型（e)。因为没有涉及法律条文、咨询、复杂情景或上下文追问。\n\n2. **置信度评估**：置信度高，因为问候语很明确，不需要更多信息。\n\n3. **关键信息提取**：没有法律概念、实体、时间地点等。\n\n4. **行动规划**：根据回答规范，保持礼貌友好，不需要调用任何工具。直接回复问候即可。\n\n检查是否有其他可能，比如用户可能后续会问问题，但当前只是打招呼，所以不需要调用检索工具。确保不提供无关信息，只做简单回应。","tool_calls":[]},"finish_reason":"stop"}]}
```

### 进行问答

**目标**

在 web UI 进行问答。

**步骤**

1. 进入 Open WebUI 应用的 UI，注册并登录；
2. 选择名为“法小助”的模型，询问一个法律问题。

**验证方法**

得到基于检索的回答，并且前端样式正确。

## 技术架构

- **数据存储**: Milvus 向量数据库
- **检索方式**: 混合检索（稀疏向量 + 密集向量）
- **服务协议**: MCP (Model-Context-Protocol)
- **模型部署**: vLLM
- **向量化**: bge-m3 (稀疏) + qwen3-embedding-0.6b (密集)
