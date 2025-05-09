# Kubernetes 部署指南

本目录包含将Milvus外部知识库API服务部署到Kubernetes集群的配置文件。

## 配置说明

### 1. ConfigMap

`dify-external-knowledge-config` ConfigMap包含所有Milvus和服务相关配置：

- **Milvus连接配置**:
  - `milvus_host`: Milvus服务主机名
  - `milvus_port`: Milvus服务端口
  - `milvus_token`: Milvus认证令牌(如果需要)
  
- **搜索配置**:
  - `search_mode`: 默认搜索模式 (hybrid/dense/sparse)
  - `dense_field`: 密集向量字段名称
  - `sparse_field`: 稀疏向量字段名称
  - `dense_metric_type`: 密集向量距离度量类型
  - `sparse_metric_type`: 稀疏向量距离度量类型
  - `hybrid_rrf_k`: RRF排序的k参数
  
- **模型配置**:
  - `use_fp16`: 是否使用FP16精度 (true/false)
  - `device`: 计算设备 (cpu/cuda)

### 2. Secret

`dify-external-knowledge-secret` Secret包含API认证密钥：

- `api_key`: 用于API认证的密钥

## 部署步骤

1. 部署Milvus（如果尚未部署）：

   ```bash
   # 使用Helm部署Milvus
   helm repo add milvus https://milvus-io.github.io/milvus-helm
   helm repo update
   helm install milvus-standalone milvus/milvus --set service.type=ClusterIP
   ```

2. 部署外部知识库API服务：

   ```bash
   # 应用ConfigMap、Secret、Deployment和Service
   kubectl apply -f deployment.yaml
   ```

3. 检查部署状态：

   ```bash
   kubectl get pods
   kubectl get services
   ```

## 自定义配置

### 修改ConfigMap

可以通过以下命令修改ConfigMap：

```bash
kubectl edit configmap dify-external-knowledge-config
```

### 修改API密钥

生成强随机密钥并更新Secret：

```bash
# 生成随机密钥
API_KEY=$(openssl rand -hex 16)
echo $API_KEY

# 更新Secret
kubectl create secret generic dify-external-knowledge-secret \
  --from-literal=api_key=$API_KEY \
  --dry-run=client -o yaml | kubectl apply -f -
```

## 访问服务

服务默认以ClusterIP类型部署，可以通过以下方式访问：

1. 使用Kubernetes集群内部DNS:
   ```
   http://dify-external-knowledge.default.svc.cluster.local/health
   ```

2. 创建Ingress（需要Ingress Controller）:
   ```bash
   kubectl apply -f ingress.yaml  # 如果有ingress.yaml配置
   ```

3. 使用端口转发进行测试:
   ```bash
   kubectl port-forward svc/dify-external-knowledge 8080:80
   ```
   然后访问: http://localhost:8080/health

## 日志和监控

查看服务日志：

```bash
# 获取Pod名称
POD_NAME=$(kubectl get pods -l app=dify-external-knowledge -o jsonpath="{.items[0].metadata.name}")

# 查看日志
kubectl logs $POD_NAME
```

## 资源调整

默认配置中，Pod请求4Gi内存和2个CPU核心，限制为8Gi内存和4个CPU核心。根据实际需要，可以在`deployment.yaml`中调整这些值。 