# 智能法律问答应用

这是一个智能法律问答应用的项目，旨在构建一个能够准确、简明回答用户法律问题的智能助手。

## 项目概述

该项目构建了一个完整的法律知识检索和问答系统，包含：

- **法律数据处理**：处理刑事案例、民事案例和法律法规数据
- **向量检索服务**：基于 MCP 协议的多种检索服务
- **模型部署**：支持多种 LLM 和嵌入模型的部署配置
- **应用集成**：与第三方应用（如 Cherry Studio）的集成指南

## 项目结构

```
law-qa/
├── agent/               # Agent 相关配置
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

## 快速开始

### 数据准备
查看 [data/README.md](./data/README.md) 了解如何准备和处理法律数据。

### 服务部署
系统包含三种核心服务，查看对应文档了解部署方法：

1. **向量数据库**：查看 [app-configs/milvus/README.md](./app-configs/milvus/README.md) 了解 Milvus 应用配置
2. **模型服务**：查看 [app-configs/vllm/README.md](./app-configs/vllm/README.md) 了解 vLLM 应用配置  
3. **检索服务**：查看 [mcp-servers/README.md](./mcp-servers/README.md) 了解 MCP 检索服务部署

### 应用集成

#### Cherry Studio 集成示例

1. **初始设置**
   - 启动 Cherry Studio App，进入 `设置` -> `MCP 服务器`
   - 如果右上角有红色叹号，点击它来安装 `UV` 和 `Bun`

2. **添加 MCP 服务器**
   - 点击 `添加服务器` -> `快速创建`
   - **名称**: 自定义名称，例如 `law-searcher`
   - **类型**: 推荐使用 `可流式传输的HTTP (streamableHttp)`
   - **URL**: 填写 VirtualService 中配置的 URL，例如 `https://home.qy.t9kcloud.cn/mcp/law-searcher/mcp/`（注意末尾斜杠）
   - **请求头**: 如需认证，添加 `Authorization=Bearer <token>`

3. **设置模型服务**
   - 点击 `设置` -> `模型服务`
   - 选择 `vLLM`
   - 填写 `API 地址`，例如 `http://localhost:8000`
   - 填写 `模型名称`，例如 `Qwen3-32B`

4. **创建助手**
   - 点击 `添加助手` -> `编辑助手`
   - 填写[提示词](./agent/system-prompt.txt)
   - 启用 MCP 服务器
   - 开始问答

## 技术架构

- **数据存储**: Milvus 向量数据库
- **检索方式**: 混合检索（稀疏向量 + 密集向量）
- **服务协议**: MCP (Model-Context-Protocol)
- **模型部署**: vLLM
- **向量化**: bge-m3 (稀疏) + qwen3-embedding-0.6b (密集)
