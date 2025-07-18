# MCP 检索服务

本目录包含基于 MCP (Model-Context-Protocol) 协议的检索服务，用于为法律助手提供多种检索能力。

## 服务概述

### law-searcher
- **功能**: 提供法条检索功能
- **覆盖数据**: 刑法、民法典等法律法规
- **检索方式**: 支持稀疏、密集、混合检索和精确查询
- **工具数量**: 8个检索工具
- **传输模式**: 支持 HTTP 和 SSE 两种模式

### case-searcher
- **功能**: 提供案例检索功能
- **覆盖数据**: 刑事案例、民事案例
- **检索方式**: 支持稀疏、密集、混合检索和精确查询
- **工具数量**: 8个检索工具
- **传输模式**: 支持 HTTP 和 SSE 两种模式

### reranker
- **功能**: 对检索结果进行重排序
- **算法**: 基于 bge-reranker-v2-m3 模型
- **用途**: 提升检索结果的相关性和准确性
- **传输模式**: 支持 HTTP 和 SSE 两种模式

## 技术架构

### MCP 协议
- **标准**: 基于 Model-Context-Protocol 标准
- **通信方式**: 支持 HTTP/JSON-RPC 和 SSE (Server-Sent Events) 两种传输模式
- **集成方式**: 可直接集成到支持 MCP 的应用中

### 传输模式

#### HTTP 模式 (streamable-http)
- **特点**: 传统的 HTTP 请求/响应模式
- **适用场景**: 标准的 MCP 客户端集成
- **路径**: `/mcp/{service-name}/`

#### SSE 模式 (Server-Sent Events)
- **特点**: 实时双向流式通信
- **适用场景**: 需要实时交互的应用
- **路径**: `/mcp/{service-name}-sse/`
- **端点**: 
  - SSE 连接：`/mcp/{service-name}-sse/sse`
  - 消息发送：`/mcp/{service-name}-sse/message`

### 检索技术
- **稀疏检索**: 基于 BM25 和关键词匹配
- **密集检索**: 基于语义向量相似度
- **混合检索**: 结合稀疏和密集检索的优势
- **精确查询**: 支持元数据过滤和条件查询

### 数据库
- **向量数据库**: Milvus
- **索引类型**: HNSW 密集索引 + Inverted 稀疏索引
- **存储**: 分离式存储，支持横向扩展

## 部署架构

### Kubernetes 部署
每个服务都提供两种部署模式：

#### HTTP 模式部署
- **配置文件**: `k8s.yaml`
- **服务名称**: `mcp-server-{service-name}`
- **路径前缀**: `/mcp/{service-name}/`

#### SSE 模式部署
- **配置文件**: `k8s-sse.yaml`
- **服务名称**: `mcp-server-{service-name}-sse`
- **路径前缀**: `/mcp/{service-name}-sse/`

每种模式都包含：
- **Deployment**: 服务实例管理
- **Service**: 内部服务发现
- **VirtualService**: 外部访问入口（Istio）
- **ConfigMap**: 环境变量配置

### 容器化
- **基础镜像**: Python 官方镜像
- **包管理**: 使用 uv 进行依赖管理
- **运行环境**: 支持 GPU 加速（可选）

## 快速开始

### 本地开发
```bash
# 进入服务目录
cd law-searcher  # 或 case-searcher / reranker

# 安装依赖
uv sync

# 启动 HTTP 模式
uv run python server.py

# 启动 SSE 模式
uv run python server.py --sse
```

### Kubernetes 部署

#### 部署 HTTP 模式服务
```bash
# 部署单个服务 (HTTP 模式)
kubectl apply -f law-searcher/k8s.yaml

# 部署所有服务 (HTTP 模式)
kubectl apply -f */k8s.yaml
```

#### 部署 SSE 模式服务
```bash
# 部署单个服务 (SSE 模式)
kubectl apply -f law-searcher/k8s-sse.yaml

# 部署所有服务 (SSE 模式)
kubectl apply -f */k8s-sse.yaml
```

## 服务配置

### 环境变量
- `MILVUS_HOST`: Milvus 服务地址
- `MILVUS_PORT`: Milvus 服务端口
- `MILVUS_USER`: Milvus 用户名（可选）
- `MILVUS_PASSWORD`: Milvus 密码（可选）
- `MCP_SECRET`: MCP 服务认证密钥（可选）

### 网络配置

#### HTTP 模式
- **内部端口**: 8000
- **健康检查**: `/health`
- **MCP 端点**: `/mcp/{service-name}/`

#### SSE 模式
- **内部端口**: 8000
- **健康检查**: `/health`
- **SSE 连接**: `/mcp/{service-name}-sse/sse`
- **消息发送**: `/mcp/{service-name}-sse/message`

## 性能调优

### 缓存策略
- **查询缓存**: 对频繁查询结果进行缓存
- **连接池**: 复用数据库连接
- **批量处理**: 支持批量检索请求

### 扩展性
- **水平扩展**: 支持多实例部署
- **负载均衡**: 通过 Kubernetes Service 实现
- **监控指标**: 提供 Prometheus 兼容的指标

## 使用示例

### Cherry Studio 集成

#### HTTP 模式集成
1. 添加 MCP 服务器
2. 配置服务 URL：`https://your-domain/mcp/{service-name}/`
3. 设置认证信息（如需要）
4. 测试连接和功能

#### SSE 模式集成
1. 添加 MCP 服务器（SSE 传输）
2. 配置 SSE URL：`https://your-domain/mcp/{service-name}-sse/sse`
3. 配置消息 URL：`https://your-domain/mcp/{service-name}-sse/message`
4. 设置认证信息（如需要）
5. 测试连接和功能

### API 调用

#### HTTP 模式
```bash
# 健康检查
curl https://your-domain/mcp/law-searcher/health

# MCP 协议调用
curl -X POST https://your-domain/mcp/law-searcher/ \
  -H "Content-Type: application/json" \
  -d '{"method": "tools/list"}'
```

#### SSE 模式
```bash
# 健康检查
curl https://your-domain/health

# SSE 连接
curl -N https://your-domain/mcp/law-searcher-sse/sse

# 发送消息
curl -X POST https://your-domain/mcp/law-searcher-sse/message \
  -H "Content-Type: application/json" \
  -d '{"method": "tools/list"}'
```

## 监控和运维

### 日志
- **格式**: 结构化 JSON 格式
- **级别**: INFO/DEBUG/ERROR
- **内容**: 请求追踪、性能指标、错误信息

### 健康检查
- **端点**: `/health`
- **检查项**: 数据库连接、服务状态
- **响应**: JSON 格式的健康状态

### 故障排查
1. 检查服务日志
2. 验证数据库连接
3. 确认网络配置
4. 测试 MCP 协议通信

## 开发指南

### 添加新工具
1. 在 `server.py` 中定义工具函数
2. 注册工具到 MCP 服务器
3. 实现检索逻辑
4. 添加单元测试

### 代码规范
- 使用 Black 进行代码格式化
- 遵循 PEP 8 编码标准
- 添加类型注解
- 编写文档字符串

## 相关文档

- [law-searcher 详细文档](./law-searcher/README.md)
- [case-searcher 详细文档](./case-searcher/README.md)
- [reranker 详细文档](./reranker/README.md)
- [MCP 协议规范](https://modelcontextprotocol.io/)

## 问题反馈

如遇到问题，请：
1. 查看服务日志
2. 检查配置信息
3. 参考相关文档
4. 提交 Issue 报告 