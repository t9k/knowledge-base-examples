# 智能法律问答 Agent

本目录包含法律智能问答的 Agent 实现，负责处理用户交互、调用 LLM 以及调用 MCP 工具。已支持三种交互方式：命令行（CLI）、WebUI、OpenAI 兼容 API。

## 目录结构与说明

- `app.py`: 统一入口，根据 `--mode` 启动 CLI/WebUI/API
- `core/`
  - `config.py`: 配置聚合（命令行/环境变量/默认值），导出 `AgentConfig`
  - `bot.py`: `create_bot(config) -> Assistant` 工厂；系统提示词与 MCP 工具装配
  - `tokenizer.py`: 分词器加载与缓存
  - `conversation.py`: 会话存储与追加（目前内存实现）
- `interfaces/`
  - `cli.py`: CLI 循环
  - `webui.py`: WebUI 启动封装
  - `openai_api.py`: FastAPI 实现的 OpenAI 兼容接口（含流式 SSE）
- `system-prompt.md`: 系统提示词参考（实际已内置于 `core/bot.py`）

## 依赖安装

确保已安装 `uv`，然后安装依赖：

```bash
# 安装 uv
# curl -LsSf https://astral.sh/uv/install.sh | sh
# source $HOME/.cargo/env

uv sync
```

## 启动方式

以下命令均在当前目录执行。

### 命令行（CLI）

```bash
uv run python app.py --mode cli \
  --model-server http://127.0.0.1:8000/v1 \
  --law-searcher --case-searcher --reranker
```

支持指令：`/history`、`/clear`、`/exit`。

### WebUI

```bash
uv run python app.py --mode webui \
  --model-server http://127.0.0.1:8000/v1 \
  --law-searcher --case-searcher --reranker
```

可选头像：`--avatar ./chatbot.png`（默认同目录）。

### OpenAI Compatible API

```bash
uv run python app.py --mode api \
  --api-host 0.0.0.0 --api-port 8001 --api-key YOUR_KEY \
  --model-server http://127.0.0.1:8000/v1 \
  --law-searcher --case-searcher --reranker
```

- 健康检查：`GET /healthz`
- 列表模型：`GET /v1/models`
- Chat Completions：`POST /v1/chat/completions`
  - 非流式示例：
    ```bash
    curl -H "Authorization: Bearer YOUR_KEY" -H "Content-Type: application/json" \
      -d '{"model":"Qwen3-32B","messages":[{"role":"user","content":"你好"}]}' \
      http://127.0.0.1:8001/v1/chat/completions
    ```
  - 流式示例（SSE）：
    ```bash
    curl -N -H "Authorization: Bearer YOUR_KEY" -H "Content-Type: application/json" \
      -d '{"model":"Qwen3-32B","stream":true,"messages":[{"role":"user","content":"你好"}]}' \
      http://127.0.0.1:8001/v1/chat/completions
    ```

## 使用 Docker 运行（OpenAI Compatible API）

已提供 `Dockerfile`，用于直接以 API 模式启动。

### 构建镜像

```bash
cd law-qa/agent
docker build -t your-registry/law-agent:api .
```

### 本地运行

```bash
docker run --rm -p 8001:8001 \
  -e CHAT_API_KEY=YOUR_KEY \
  -e CHAT_BASE_URL=http://host.docker.internal:8000/v1 \
  -e LAW_SEARCHER_URL=https://home.qy.t9kcloud.cn/mcp/law-searcher/mcp/ \
  -e CASE_SEARCHER_URL=https://home.qy.t9kcloud.cn/mcp/case-searcher/mcp/ \
  -e RERANKER_URL=https://home.qy.t9kcloud.cn/mcp/reranker/mcp/ \
  your-registry/law-agent:api
```

默认命令会以 `--mode api --api-host 0.0.0.0 --api-port 8001` 启动；若需自定义可覆盖 `CMD` 或追加参数。

### 健康检查与调用

```bash
curl http://127.0.0.1:8001/healthz
curl -H "Authorization: Bearer YOUR_KEY" -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-32B","messages":[{"role":"user","content":"你好"}]}' \
  http://127.0.0.1:8001/v1/chat/completions
```

## 在 Kubernetes 部署（OpenAI Compatible API）

已提供示例清单 `k8s.yaml`，包含 `Deployment`、`Service` 和 `Secret`，需要修改以下环境变量：

   - `CHAT_BASE_URL`：LLM 的 vLLM 服务端点
   - `LAW_SEARCHER_URL`、`CASE_SEARCHER_URL`、`RERANKER_URL`：MCP 服务地址
   - `CHAT_API_KEY`：在 `Secret` 中设置

应用清单：

```bash
kubectl apply -f k8s.yaml
```

服务暴露：

- 集群内访问：`http://law-agent.default.svc.cluster.local:8001`
- 若需公网访问，可为 `Service` 增加 `Ingress`（按你的 Ingress 控制器配置）。

## 常用参数

- 模式与基础
  - `--mode {cli|webui|api}`：选择交互方式
  - `--model`：模型名称（默认 `Qwen3-32B`）
  - `--model-server`：模型服务地址（默认 `http://127.0.0.1:8000/v1`）
  - `--tokenizer-path`：分词器路径（默认 `Qwen/Qwen3-32B`）
  - `--max-tokens`：对话历史最大 Token（默认 `10000`）
- MCP 工具开关与地址
  - `--law-searcher` / `--law-searcher-url`
  - `--case-searcher` / `--case-searcher-url`
  - `--reranker` / `--reranker-url`
- API 专属
  - `--api-host` / `--api-port`
  - `--api-key`：`Authorization: Bearer <API_KEY>` 鉴权
  - `--allow-cors`：允许跨域
- WebUI
  - `--avatar`：头像路径（默认 `./chatbot.png`）

## 环境变量（可选）

- `CHAT_MODEL`, `CHAT_BASE_URL`, `AGENT_TOKENIZER_PATH`, `AGENT_MAX_TOKENS`
- `LAW_SEARCHER_URL`, `CASE_SEARCHER_URL`, `RERANKER_URL`
- `AGENT_API_HOST`, `AGENT_API_PORT`, `CHAT_API_KEY`
- `AGENT_AVATAR_PATH`

示例：

```bash
export LAW_SEARCHER_URL=https://home.qy.t9kcloud.cn/mcp/law-searcher/mcp/
export CASE_SEARCHER_URL=https://home.qy.t9kcloud.cn/mcp/case-searcher/mcp/
export RERANKER_URL=https://home.qy.t9kcloud.cn/mcp/reranker/mcp/
uv run python app.py --mode api --api-key YOUR_KEY
```

## 说明

- 本项目内置系统提示词于 `core/bot.py`，`system-prompt.md` 作为参考与版本化记录。
- OpenAI API 兼容实现已支持非流式与流式（SSE），可直接被常见 OpenAI SDK/客户端调用。
