# Milvus数据库配置
MILVUS_CONFIG = {
    "host": "localhost",  # Milvus服务器主机名或IP地址
    "port": "19530",      # Milvus服务器端口
    "token": "",          # Milvus身份验证令牌，默认留空表示不使用身份验证
    # 默认检索模式: dense(密集向量), sparse(稀疏向量), hybrid(混合检索)
    "search_mode": "hybrid",
    # 密集向量检索配置
    "dense_search": {
        "dense_field": "dense_vector",  # 密集向量字段名
        "params": {
            "metric_type": "COSINE",    # 距离度量类型: COSINE, L2, IP
            "params": {}                # 索引特定参数
        }
    },
    # 稀疏向量检索配置
    "sparse_search": {
        "sparse_field": "sparse_vector",  # 稀疏向量字段名
        "params": {
            "metric_type": "IP",          # 距离度量类型，稀疏向量通常使用IP(内积)
            "params": {}                  # 索引特定参数
        }
    },
    # 混合检索配置
    "hybrid_search": {
        "dense_field": "dense_vector",    # 密集向量字段名
        "sparse_field": "sparse_vector",  # 稀疏向量字段名
        "dense_params": {
            "metric_type": "COSINE",      # 密集向量度量类型
            "params": {}                  # 密集向量索引特定参数
        },
        "sparse_params": {
            "metric_type": "IP",          # 稀疏向量度量类型
            "params": {}                  # 稀疏向量索引特定参数
        },
        "rrf_k": 60  # RRF排序的k参数，值越大表示排名影响越小
    },
    # 嵌入模型配置
    "embedding": {
        "use_fp16": False,  # 是否使用FP16精度加速计算（需要GPU支持）
        "device": "cpu"     # 运行设备: "cpu" 或 "cuda:0" 等GPU设备
    }
}

# API密钥配置，留空列表表示不需要身份验证
API_KEYS = [
    # 将您的API密钥添加到此列表
    "t9k-12345",  # 示例API密钥，应在生产环境中替换为强随机密钥
]
