# 法律智能助手 Agent

本目录包含法律智能助手的核心 Agent 实现，负责处理用户交互、调用 MCP 工具以及与语言模型通信。

## 文件说明

- `agent.py`: Agent 的主程序。它使用 `qwen_agent` 库构建，可以启动一个命令行界面（CLI）或 Web UI 与用户交互。
- `system-prompt.txt`: 定义了 Agent 的角色、行为准则和思考框架。Agent 在启动时会加载此文件作为其核心指令。

## 功能特点

- **双模式交互**: 支持通过传统的命令行或更友好的 Web UI 与 Agent 聊天。
- **模块化工具调用**: 可通过命令行参数动态启用不同的 MCP 服务（`law-searcher`, `case-searcher`, `reranker`），实现灵活的知识检索和处理能力。
- **可配置模型**: 支持通过参数指定后端的大语言模型及其服务地址。
- **智能上下文管理**: 自动管理对话历史，确保在不超过模型 Token 限制的前提下保持对话的连贯性。

## 快速开始

### 依赖安装

确保已安装 `uv`，然后安装项目所需的依赖：

```bash
# 安装 uv
# curl -LsSf https://astral.sh/uv/install.sh | sh
# source $HOME/.cargo/env

uv sync
```

### 启动 Agent

你可以根据需要组合不同的参数来启动 Agent。

**1. 启动基本的命令行 Agent（不带任何工具）**

```bash
python agent.py --model-server <your-llm-api-endpoint>
```

**2. 启用特定的 MCP 检索工具**

- **启用法条检索**:
  ```bash
  python agent.py --model-server <your-llm-api-endpoint> --law-searcher
  ```

- **启用案例检索和重排服务**:
  ```bash
  python agent.py --model-server <your-llm-api-endpoint> --case-searcher --reranker
  ```

- **启用所有工具**:
  ```bash
  python agent.py --model-server <your-llm-api-endpoint> --law-searcher --case-searcher --reranker
  ```

**3. 启动 Web UI**

在任何启动命令后添加 `--webui` 参数即可启动 Web 界面。

```bash
python agent.py --model-server <your-llm-api-endpoint> --law-searcher --case-searcher --reranker --webui
```

### 可选参数

- `--model`: 指定使用的大语言模型名称 (默认: `Qwen3-32B`)。
- `--model-server`: 指定语言模型服务的 API 地址 (例如: `http://127.0.0.1:8000/v1`)。
- `--tokenizer-path`: 指定分词器模型的路径 (默认: `Qwen/Qwen3-32B`)。
- `--max-tokens`: 指定对话历史的最大 Token 数量 (默认: `10000`)。
