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

### OpenAI 兼容 API

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

- `AGENT_MODEL`, `AGENT_MODEL_SERVER`, `AGENT_TOKENIZER_PATH`, `AGENT_MAX_TOKENS`
- `LAW_SEARCHER_URL`, `CASE_SEARCHER_URL`, `RERANKER_URL`
- `AGENT_API_HOST`, `AGENT_API_PORT`, `AGENT_API_KEY`
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
