from flask import Flask, request, jsonify, abort
import logging
from pymilvus import connections, Collection, utility
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus import AnnSearchRequest, RRFRanker
import json
import os
import sys
from functools import wraps

from config import MILVUS_CONFIG, OTHER_CONFIG, API_KEYS

# 设置环境变量来控制更详细的调试输出
debug_mode = OTHER_CONFIG["debug_mode"]
if debug_mode:
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)])
else:
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logging.info(f"Application starting with debug_mode={debug_mode}")

app = Flask(__name__)

# Error codes as defined in the API spec
ERROR_CODES = {
    "INVALID_AUTH_FORMAT": 1001,
    "AUTH_FAILED": 1002,
    "KNOWLEDGE_NOT_FOUND": 2001,
    "MILVUS_CONNECTION_ERROR": 3001
}


# 身份验证装饰器
def require_api_key(f):

    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')

        # 如果没有配置API密钥，跳过验证
        if not API_KEYS:
            return f(*args, **kwargs)

        # 检查Authorization头部格式
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({
                "error_code":
                ERROR_CODES["INVALID_AUTH_FORMAT"],
                "error_msg":
                "Invalid Authorization header format. Expected format is 'Bearer <api-key>'."
            }), 403

        # 提取API密钥
        api_key = auth_header.split(' ')[1]

        # 验证API密钥
        if api_key not in API_KEYS:
            logging.warning(
                f"Authentication failed with API key: {api_key[:5]}...")
            return jsonify({
                "error_code":
                ERROR_CODES["AUTH_FAILED"],
                "error_msg":
                "Authorization failed. Invalid API key."
            }), 403

        return f(*args, **kwargs)

    return decorated_function


# 创建一个请求/响应日志中间件
@app.before_request
def log_request_info():
    """记录请求详情"""
    if request.method in ['POST', 'PUT', 'PATCH']:
        try:
            # 仅记录JSON请求体
            if request.is_json:
                request_data = request.get_json()
                logging.info(
                    f"Request body: {json.dumps(request_data, ensure_ascii=False)}"
                )
            else:
                logging.info(f"Request content-type: {request.content_type}")
        except Exception as e:
            logging.error(f"Error logging request: {str(e)}")


@app.after_request
def log_response_info(response):
    """记录响应详情"""
    try:
        response_data = response.get_data()
        if response.status_code >= 400:  # 只记录错误响应详情
            # 尝试解析JSON数据
            try:
                if response.content_type == 'application/json':
                    response_json = json.loads(response_data.decode('utf-8'))
                    logging.error(
                        f"Error response: {json.dumps(response_json, ensure_ascii=False)}"
                    )
                else:
                    logging.error(
                        f"Error response: {response_data.decode('utf-8')}")
            except Exception as e:
                logging.error(f"Error response (raw): {response_data}")
        logging.info(f"Response status: {response.status_code}")
    except Exception as e:
        logging.error(f"Error logging response: {str(e)}")
    return response


# 初始化嵌入模型
device = OTHER_CONFIG["embedding"]["device"]
use_fp16 = OTHER_CONFIG["embedding"]["use_fp16"]
if device == "gcu":
    import torch
    import torch_gcu
    embedding_function = BGEM3EmbeddingFunction(use_fp16=use_fp16,
                                                device="gcu")
elif device == "cpu":
    embedding_function = BGEM3EmbeddingFunction(use_fp16=use_fp16,
                                                device="cpu")
else:
    raise ValueError(f"Unsupported embedding device: {device}")
logging.info(f"BGEM3 embedding model initialized on {device}")


# 健康检查端点 - 不需要认证
@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点，不需要认证"""
    status = {
        "status": "ok",
        "version": "1.0.0",
        "auth_enabled": len(API_KEYS) > 0
    }
    return jsonify(status)


def translate_operator(operator):
    """
    将API的比较运算符转换为Milvus过滤语法的运算符
    """
    operator_mapping = {
        "contains": "LIKE",  # Milvus使用LIKE "%value%"
        "not contains": "LIKE",  # Milvus使用LIKE "%value%"
        "start with": "LIKE",  # Milvus使用LIKE "value%"
        "end with": "LIKE",  # Milvus使用LIKE "%value"
        "is": "==",
        "is not": "!=",
        "empty": "IS NULL",
        "not empty": "IS NOT NULL",
        "=": "==",
        "≠": "!=",
        ">": ">",
        "<": "<",
        "≥": ">=",
        "≤": "<=",
        "before": "<",  # 日期比较
        "after": ">"  # 日期比较
    }
    return operator_mapping.get(operator, operator)


def build_milvus_filter(metadata_condition):
    """
    将API的metadata_condition转换为Milvus过滤表达式
    """
    if not metadata_condition or "conditions" not in metadata_condition:
        return None

    conditions = metadata_condition["conditions"]
    if not conditions:
        return None

    logical_op = metadata_condition.get("logical_operator", "and").upper()

    filter_parts = []
    for condition in conditions:
        name = condition["name"]
        dify_op = condition["comparison_operator"]
        if not name:
            continue

        field_name = name[0]  # 使用第一个元素作为字段名
        op = translate_operator(dify_op)

        # 处理不需要值的运算符
        if op in ["IS NULL", "IS NOT NULL"]:
            filter_parts.append(f"{field_name} {op}")
            continue

        value = condition.get("value", "")

        # 根据运算符格式化值
        if op == "LIKE":
            if dify_op in ["contains", "not contains"]:
                formatted_value = f'"%{value}%"'
            elif dify_op == "start with":
                formatted_value = f'"{value}%"'
            elif dify_op == "end with":
                formatted_value = f'"%{value}"'
            else:
                formatted_value = f'"{value}"'
        elif isinstance(value, str):
            formatted_value = f'"{value}"'
        else:
            formatted_value = str(value)

        if dify_op in ["not contains"]:
            filter_parts.append(f"NOT ({field_name} {op} {formatted_value})")
        else:
            filter_parts.append(f"{field_name} {op} {formatted_value}")

    # 使用logical_op连接所有条件
    filter_expr = f" {logical_op} ".join(filter_parts)
    return filter_expr


def perform_search(collection,
                   query_text,
                   search_mode,
                   limit=10,
                   expr=None,
                   output_fields=None):
    """
    通用搜索函数，根据搜索模式执行不同的搜索策略
    """
    # 生成查询嵌入
    query_embeddings = embedding_function([query_text])

    if search_mode == "hybrid":
        # 混合搜索
        hybrid_config = MILVUS_CONFIG["hybrid_search"]

        # 创建密集向量搜索请求
        dense_req = AnnSearchRequest(query_embeddings["dense"],
                                     hybrid_config["dense_field"],
                                     hybrid_config["dense_params"],
                                     limit=limit,
                                     expr=expr)

        # 创建稀疏向量搜索请求
        sparse_req = AnnSearchRequest([query_embeddings["sparse"]],
                                      hybrid_config["sparse_field"],
                                      hybrid_config["sparse_params"],
                                      limit=limit,
                                      expr=expr)

        # 使用RRF重排序
        rerank = RRFRanker(hybrid_config["rrf_k"])

        # 执行混合搜索
        results = collection.hybrid_search([sparse_req, dense_req],
                                           rerank=rerank,
                                           limit=limit,
                                           output_fields=output_fields)[0]

    elif search_mode == "dense":
        # 密集向量搜索
        dense_config = MILVUS_CONFIG["dense_search"]

        results = collection.search(data=query_embeddings["dense"],
                                    anns_field=dense_config["dense_field"],
                                    param=dense_config["params"],
                                    limit=limit,
                                    expr=expr,
                                    output_fields=output_fields)[0]

    elif search_mode == "sparse":
        # 稀疏向量搜索
        sparse_config = MILVUS_CONFIG["sparse_search"]

        results = collection.search(data=[query_embeddings["sparse"]],
                                    anns_field=sparse_config["sparse_field"],
                                    param=sparse_config["params"],
                                    limit=limit,
                                    expr=expr,
                                    output_fields=output_fields)[0]
    else:
        raise ValueError(f"Unsupported search mode: {search_mode}")

    return results


@app.route('/retrieval', methods=['POST'])
@require_api_key
def retrieval():
    try:
        # Get JSON data from request
        data = request.get_json()

        # 验证请求参数
        if not data:
            logging.error("Empty request body")
            return jsonify({
                "error_code": ERROR_CODES["KNOWLEDGE_NOT_FOUND"],
                "error_msg": "Request body is empty"
            }), 400

        # 验证必填字段
        required_fields = ["knowledge_id", "query", "retrieval_setting"]
        missing_fields = [
            field for field in required_fields if field not in data
        ]
        if missing_fields:
            return jsonify({
                "error_code":
                ERROR_CODES["KNOWLEDGE_NOT_FOUND"],
                "error_msg":
                f"Missing required fields: {', '.join(missing_fields)}"
            }), 400

        # 验证retrieval_setting
        retrieval_setting = data.get("retrieval_setting", {})
        if not isinstance(retrieval_setting, dict):
            return jsonify({
                "error_code": ERROR_CODES["KNOWLEDGE_NOT_FOUND"],
                "error_msg": "retrieval_setting must be an object"
            }), 400

        # 验证top_k和score_threshold
        if "top_k" not in retrieval_setting or "score_threshold" not in retrieval_setting:
            return jsonify({
                "error_code":
                ERROR_CODES["KNOWLEDGE_NOT_FOUND"],
                "error_msg":
                "Missing required fields in retrieval_setting: top_k and score_threshold"
            }), 400

        # 验证top_k是整数
        if not isinstance(retrieval_setting["top_k"], int):
            return jsonify({
                "error_code": ERROR_CODES["KNOWLEDGE_NOT_FOUND"],
                "error_msg": "top_k must be an integer"
            }), 400

        # 验证score_threshold是0到1之间的浮点数
        if not isinstance(retrieval_setting["score_threshold"], (int, float)) or \
           retrieval_setting["score_threshold"] < 0 or retrieval_setting["score_threshold"] > 1:
            return jsonify({
                "error_code":
                ERROR_CODES["KNOWLEDGE_NOT_FOUND"],
                "error_msg":
                "score_threshold must be a float between 0 and 1"
            }), 400

        # 获取搜索模式
        search_mode = data.get("search_mode", MILVUS_CONFIG["search_mode"])
        # 验证搜索模式
        if search_mode not in ["dense", "sparse", "hybrid"]:
            return jsonify({
                "error_code":
                ERROR_CODES["KNOWLEDGE_NOT_FOUND"],
                "error_msg":
                "Invalid search_mode. Must be one of: dense, sparse, hybrid"
            }), 400

        # 验证并处理metadata_condition
        milvus_filter = None
        if "metadata_condition" in data and data[
                "metadata_condition"] is not None:
            metadata_condition = data["metadata_condition"]

            # 基本类型验证
            if not isinstance(metadata_condition, dict):
                return jsonify({
                    "error_code":
                    ERROR_CODES["KNOWLEDGE_NOT_FOUND"],
                    "error_msg":
                    "metadata_condition must be an object"
                }), 400

            # 验证conditions字段
            if "conditions" not in metadata_condition:
                return jsonify({
                    "error_code":
                    ERROR_CODES["KNOWLEDGE_NOT_FOUND"],
                    "error_msg":
                    "Missing required field in metadata_condition: conditions"
                }), 400

            if not isinstance(metadata_condition["conditions"], list):
                return jsonify({
                    "error_code": ERROR_CODES["KNOWLEDGE_NOT_FOUND"],
                    "error_msg": "conditions must be an array"
                }), 400

            # 验证logical_operator
            if "logical_operator" in metadata_condition and \
               metadata_condition["logical_operator"] not in ["and", "or"]:
                return jsonify({
                    "error_code":
                    ERROR_CODES["KNOWLEDGE_NOT_FOUND"],
                    "error_msg":
                    "logical_operator must be 'and' or 'or'"
                }), 400

            # 验证每个条件
            valid_operators = [
                "contains", "not contains", "start with", "end with", "is",
                "is not", "empty", "not empty", "=", "≠", ">", "<", "≥", "≤",
                "before", "after"
            ]
            value_optional_operators = [
                "empty", "not empty", "null", "not null"
            ]

            for condition in metadata_condition["conditions"]:
                if not isinstance(condition, dict):
                    return jsonify({
                        "error_code":
                        ERROR_CODES["KNOWLEDGE_NOT_FOUND"],
                        "error_msg":
                        "Each condition must be an object"
                    }), 400

                # 验证必填字段
                if "name" not in condition or "comparison_operator" not in condition:
                    return jsonify({
                        "error_code":
                        ERROR_CODES["KNOWLEDGE_NOT_FOUND"],
                        "error_msg":
                        "Each condition must have 'name' and 'comparison_operator' fields"
                    }), 400

                # 验证name是字符串数组
                if not isinstance(condition["name"], list):
                    return jsonify({
                        "error_code":
                        ERROR_CODES["KNOWLEDGE_NOT_FOUND"],
                        "error_msg":
                        "name must be an array of strings"
                    }), 400

                # 验证comparison_operator
                if condition["comparison_operator"] not in valid_operators:
                    return jsonify({
                        "error_code":
                        ERROR_CODES["KNOWLEDGE_NOT_FOUND"],
                        "error_msg":
                        f"Invalid comparison_operator: {condition['comparison_operator']}"
                    }), 400

                # 验证value字段（除了empty/not empty运算符外都需要）
                if condition[
                        "comparison_operator"] not in value_optional_operators and "value" not in condition:
                    return jsonify({
                        "error_code":
                        ERROR_CODES["KNOWLEDGE_NOT_FOUND"],
                        "error_msg":
                        f"value field is required for operator: {condition['comparison_operator']}"
                    }), 400

            # 构建Milvus过滤表达式
            milvus_filter = build_milvus_filter(metadata_condition)
            logging.info(f"Generated Milvus filter: {milvus_filter}")

        # 连接到 Milvus
        database_name, collection_name = data.get("knowledge_id").split("/")
        uri = f"{MILVUS_CONFIG['host']}:{MILVUS_CONFIG['port']}"
        connections.connect(uri=uri, token=MILVUS_CONFIG["token"], db_name=database_name)
        logging.info("Connected to Milvus")

        # 获取请求参数
        query_text = data.get("query")
        top_k = retrieval_setting.get("top_k", 10)

        # 检查集合是否存在
        if not utility.has_collection(collection_name):
            logging.error(f"Knowledge base '{collection_name}' not found")
            return jsonify({
                "error_code":
                ERROR_CODES["KNOWLEDGE_NOT_FOUND"],
                "error_msg":
                f"Knowledge base '{collection_name}' not found"
            }), 404

        # 加载集合
        try:
            collection = Collection(collection_name)
            collection.load()

            scenario = OTHER_CONFIG["scenario"]

            # 需要输出的字段
            if scenario == "law":
                output_fields = [
                    "chunk", "law", "part", "chapter", "section", "article",
                    "article_amended"
                ]
            elif scenario == "criminal-cases":
                output_fields = [
                    "chunk", "relevant_articles", "accusation",
                    "punish_of_money", "criminals", "imprisonment",
                    "life_imprisonment", "death_penalty"
                ]

            # 执行搜索
            results = perform_search(collection,
                                     query_text,
                                     search_mode,
                                     limit=top_k,
                                     expr=milvus_filter,
                                     output_fields=output_fields)

            # 格式化搜索结果
            records = []
            for hit in results:
                # 计算分数
                if search_mode == "hybrid":
                    score = hit.score / (
                        2 / (MILVUS_CONFIG["hybrid_search"]["rrf_k"] + 1)
                    )  # 归一化到0-1范围
                else:
                    # 根据搜索模式和度量类型转换分数
                    if search_mode == "dense":
                        metric_type = MILVUS_CONFIG["dense_search"]["params"][
                            "metric_type"].upper()
                    else:  # sparse
                        metric_type = MILVUS_CONFIG["sparse_search"]["params"][
                            "metric_type"].upper()

                    if metric_type == "L2":
                        score = max(0.0, min(1.0, 1.0 / (1.0 + hit.score)))
                    elif metric_type == "IP":
                        score = max(0.0, min(1.0, hit.score))
                    elif metric_type == "COSINE":
                        score = max(0.0, min(1.0, hit.score))
                    else:
                        logging.error(
                            f"Unsupported metric type: {metric_type}")
                        return jsonify({
                            "error_code":
                            ERROR_CODES["KNOWLEDGE_NOT_FOUND"],
                            "error_msg":
                            f"Unsupported metric type: {metric_type}"
                        }), 400

                # 应用score_threshold
                if score < retrieval_setting["score_threshold"]:
                    continue

                # 准备记录
                if scenario == "law":
                    metadata = {
                        "law": hit.law,
                        "part": hit.part,
                        "chapter": hit.chapter,
                        "section": hit.section,
                        "article": hit.article,
                        "article_amended": hit.article_amended
                    }
                elif scenario == "criminal-cases":
                    metadata = {
                        "relevant_articles": list(hit.relevant_articles),
                        "accusation": list(hit.accusation),
                        "punish_of_money": hit.punish_of_money,
                        "criminals": list(hit.criminals),
                        "imprisonment": hit.imprisonment,
                        "life_imprisonment": hit.life_imprisonment,
                        "death_penalty": hit.death_penalty
                    }

                content = hit.chunk
                if OTHER_CONFIG["result"]["include_metadata"]:
                    content += "\n\n" + json.dumps(metadata,
                                                   ensure_ascii=False)

                record = {
                    "score": score,
                    "title": hit.chunk_id,
                    "content": content,
                    "metadata": metadata,
                }

                records.append(record)

            return jsonify({"records": records})

        except Exception as e:
            logging.error(f"Error querying Milvus: {str(e)}", exc_info=True)
            # 当Milvus查询出错，但API规范要求返回结果时，返回空记录
            return jsonify({"records": []})

    except Exception as e:
        logging.error(f"Internal server error: {str(e)}", exc_info=True)
        return jsonify({
            "error_code": ERROR_CODES["KNOWLEDGE_NOT_FOUND"],
            "error_msg": "Internal server error"
        }), 500


@app.errorhandler(403)
def forbidden(e):
    return jsonify({
        "error_code": ERROR_CODES["AUTH_FAILED"],
        "error_msg": "Authorization failed"
    }), 403


@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        "error_code": ERROR_CODES["KNOWLEDGE_NOT_FOUND"],
        "error_msg": "Internal server error"
    }), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001)
