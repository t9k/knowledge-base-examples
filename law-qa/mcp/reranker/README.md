# 重排序 MCP Server

这是一个基于 MCP (Model Context Protocol) 的重排序服务，专门用于对搜索结果进行重新排序。该服务接收其他搜索工具的结果，并使用重排序模型对结果进行重新排序，提高检索结果的相关性。

## 功能

- 实现了完整的 MCP 服务器协议，可与支持 MCP 的客户端集成
- 支持对任何格式的搜索结果进行重新排序
- 集成重排序模型API，提高检索结果的质量
- 支持健康检查端点
- 支持 sse 和 streamable-http 传输模式
- 可配置的身份认证（JWT Bearer Token）

### 传输模式

#### HTTP 模式 (streamable-http)
- **特点**: 传统的 HTTP 请求/响应模式
- **适用场景**: 标准的 MCP 客户端集成
- **路径**: `/mcp/reranker/`

#### SSE 模式 (Server-Sent Events)
- **特点**: 实时双向流式通信
- **适用场景**: 需要实时交互的应用
- **路径**: `/mcp/reranker-sse/`
- **端点**: 
  - SSE 连接：`/mcp/reranker-sse/sse`
  - 消息发送：`/mcp/reranker-sse/message`

### 身份认证

当 `ENABLE_AUTH=true` 时，服务器会启用JWT Bearer Token认证：
- 服务启动时会动态生成RSA密钥对
- 生成30天有效期的JWT token
- 客户端需要在请求头中包含 `Authorization: Bearer <token>`

当 `ENABLE_AUTH=false` 时，服务器不需要任何认证即可访问。

## 部署

### 在 Kubernetes 部署

服务提供两种部署模式：

#### HTTP 模式部署
修改 `k8s.yaml` 中的 ConfigMap 和 VirtualService 配置，然后执行以下命令进行部署：

```bash
kubectl apply -f k8s.yaml
```

#### SSE 模式部署
修改 `k8s-sse.yaml` 中的 ConfigMap 和 VirtualService 配置，然后执行以下命令进行部署：

```bash
kubectl apply -f k8s-sse.yaml
```

### 在本地部署

1. 安装依赖：

```bash
pip install fastmcp>=2.8.1 requests>=2.31.0 python-dotenv>=1.0.0 uvicorn>=0.24.0
```

2. 创建 `.env` 文件：

```env
# 重排序服务配置
RERANKER_BASE_URL=http://localhost:8001
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# 身份认证配置 (true/false)
ENABLE_AUTH=false
```

3. 启动服务：

```bash
# 启动 HTTP 模式 (streamable-http)
python server.py

# 启动 SSE 模式 (Server-Sent Events)
python server.py --sse
```

## 重排序功能

服务集成了重排序模型API，用于对搜索结果进行重新排序。

### 重排序工作流程

1. 接收来自其他搜索工具的结果字符串
2. 解析搜索结果，提取查询文本和文档内容
3. 调用重排序API对文档进行重新排序
4. 返回按相关性评分排序的结果

## MCP 工具列表

1. `rerank`：对搜索结果进行重新排序。
