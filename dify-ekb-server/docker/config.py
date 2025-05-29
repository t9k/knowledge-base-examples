import os

# 加载环境变量配置
MILVUS_CONFIG = {
    "host": os.environ.get("MILVUS_HOST", "localhost"),
    "port": os.environ.get("MILVUS_PORT", "19530"),
    "token": os.environ.get("MILVUS_TOKEN", ""),
    # 默认检索模式: dense(密集向量), sparse(稀疏向量), hybrid(混合检索)
    "search_mode": os.environ.get("SEARCH_MODE", "hybrid"),
    # 密集向量检索配置
    "dense_search": {
        "dense_field": os.environ.get("DENSE_FIELD", "dense_vector"),
        "params": {
            "metric_type": os.environ.get("DENSE_METRIC_TYPE", "COSINE"),
            "params": {}
        }
    },
    # 稀疏向量检索配置
    "sparse_search": {
        "sparse_field": os.environ.get("SPARSE_FIELD", "sparse_vector"),
        "params": {
            "metric_type": os.environ.get("SPARSE_METRIC_TYPE", "IP"),
            "params": {}
        }
    },
    # 混合检索配置
    "hybrid_search": {
        "dense_field": os.environ.get("DENSE_FIELD", "dense_vector"),
        "sparse_field": os.environ.get("SPARSE_FIELD", "sparse_vector"),
        "dense_params": {
            "metric_type": os.environ.get("DENSE_METRIC_TYPE", "COSINE"),
            "params": {}
        },
        "sparse_params": {
            "metric_type": os.environ.get("SPARSE_METRIC_TYPE", "IP"),
            "params": {}
        },
        "rrf_k": int(os.environ.get("HYBRID_RRF_K", "60"))
    }
}

# 加载其他配置
OTHER_CONFIG = {
    # 嵌入模型配置
    "embedding": {
        "use_fp16": os.environ.get("USE_FP16", "false").lower() == "true",
        "device": os.environ.get("DEVICE", "cpu")
    },
    # 检索结果配置
    "result": {
        "include_metadata":
        os.environ.get("INCLUDE_METADATA", "true").lower() == "true"
    },
    # 调试模式
    "debug_mode": os.environ.get("DEBUG_MODE", "false").lower() == "true",
    # 场景配置
    "scenario": os.environ.get("SCENARIO")
}

# API密钥配置，从环境变量加载
API_KEYS = []
if os.environ.get("API_KEY"):
    API_KEYS.append(os.environ.get("API_KEY"))
