import os
import argparse
import json
import logging
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from pymilvus import connections, Collection, AnnSearchRequest, RRFRanker
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from openai import OpenAI
import httpx
from importlib import import_module
from importlib import util as importlib_util


logger = logging.getLogger("law-searcher.search")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.INFO)
    _handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(_handler)
logger.propagate = False


def _preview(value: Any, max_len: int = 2000) -> str:
    try:
        if isinstance(value, str):
            s = value
        else:
            s = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        s = str(value)
    if len(s) > max_len:
        return f"{s[:max_len]}... (truncated, total_length={len(s)})"
    return s


def format_grouped_sources(results: List[Dict[str, Any]]) -> str:
    """将检索结果按 law+part+chapter+section 分组为 <source> 块。"""
    groups: Dict[str, Dict[str, List[Any]]] = {}

    for item in results:
        law = item.get("law")
        part = item.get("part")
        chapter = item.get("chapter")
        section = item.get("section")
        key_parts = [p for p in [law, part, chapter, section] if p]
        key = " ".join(key_parts) if key_parts else ""

        chunk_text = item.get("chunk")
        distance = item.get("distance")
        if key not in groups:
            groups[key] = {"document": [], "distances": []}
        if chunk_text is not None:
            groups[key]["document"].append(chunk_text)
        if distance is not None:
            groups[key]["distances"].append(distance)

    output_parts: List[str] = []
    for i, (name, payload) in enumerate(groups.items(), start=1):
        block = {
            "source": {"name": name},
            "document": payload["document"],
            "distances": payload["distances"],
        }
        block_str = json.dumps(block, ensure_ascii=False, indent=4)
        output_parts.append(f'<source id="{i}">\n{block_str}\n</source>')

    prompt = (
        "如需在回答中引用检索结果，请使用行内引用格式 [n]，对应 <source id=\"n\"></source>。\n\n"
        "注意：同一个 <source id=\"n\"></source> 内可能包含多篇 document，引用任意一篇时都统一写作 [n]。\n\n"
        "行内引用示例：“根据研究，该方法可提升 20% 的效率 [1]。”"
    )
    return "\n".join(output_parts) + "\n" + prompt


# 设备选择顺序：CUDA(GPU) -> GCU(Enflame) -> CPU
def _select_device() -> str:
    device = "cpu"
    try:
        if importlib_util.find_spec("torch") is None:
            raise Exception("torch not installed")
        _torch = import_module("torch")
        if hasattr(_torch, "cuda") and _torch.cuda.is_available():
            device = "cuda"
            logger.info("Detected NVIDIA CUDA GPU, using device 'cuda'")
        else:
            raise Exception("CUDA not available")
    except Exception as e:
        logger.info("CUDA not available: %s", e)
        try:
            if importlib_util.find_spec("torch") is None:
                raise Exception("torch not installed")
            _gcu_mod_name = "torch" + "_gcu"
            if importlib_util.find_spec(_gcu_mod_name) is None:
                raise Exception("torch_gcu not installed")
            import_module("torch")
            import_module(_gcu_mod_name)
            device = "gcu"
            logger.info("Detected Enflame GCU, using device 'gcu'")
        except Exception as eg:
            logger.info("GCU not available: %s, falling back to CPU", eg)
            device = "cpu"
    return device


# 初始化稀疏向量函数（BGE-M3）
_DEVICE = _select_device()
ef = BGEM3EmbeddingFunction(use_fp16=False, device=_DEVICE)


class MilvusConnector:
    def __init__(
        self,
        uri: str,
        token: Optional[str],
        db_name: str,
        law_collection_name: str,
        embedding_base_url: str,
        embedding_model: str,
        reranker_base_url: Optional[str],
        reranker_model: Optional[str],
    ) -> None:
        self.uri = uri
        self.token = token
        connections.connect(uri=uri, token=token, db_name=db_name)
        self.law_collection = Collection(law_collection_name)
        self.check_collections()
        self.law_collection.load()

        self.output_fields = [
            "chunk",
            "law",
            "part",
            "chapter",
            "section",
            "article",
        ]

        self.embedding_client = OpenAI(base_url=embedding_base_url, api_key="dummy")
        self.embedding_model = embedding_model
        self.reranker_base_url = reranker_base_url
        self.reranker_model = reranker_model

    def check_collections(self) -> None:
        expected_fields = [
            {"name": "chunk_id", "type": 21},  # VARCHAR
            {"name": "chunk", "type": 21},  # VARCHAR
            {"name": "law", "type": 21},  # VARCHAR
            {"name": "part", "type": 21},  # VARCHAR
            {"name": "chapter", "type": 21},  # VARCHAR
            {"name": "section", "type": 21},  # VARCHAR
            {"name": "article", "type": 5},  # INT64
            {"name": "article_amended", "type": 5},  # INT64
            {"name": "sparse_vector", "type": 104},  # SPARSE_FLOAT_VECTOR
            {"name": "dense_vector", "type": 101},  # FLOAT_VECTOR
        ]

        for collection in [self.law_collection]:
            schema = collection.describe()
            actual_fields = schema["fields"]
            for expected_field in expected_fields:
                found = False
                for actual_field in actual_fields:
                    if actual_field["name"] == expected_field["name"]:
                        found = True
                        if actual_field["type"].value != expected_field["type"]:
                            raise ValueError(
                                f"Field {expected_field['name']} has wrong type in {schema['collection_name']} collection. "
                                f"Expected {expected_field['type']}, got {actual_field['type'].value}"
                            )
                        break
                if not found:
                    raise ValueError(
                        f"Missing field {expected_field['name']} in {schema['collection_name']} collection"
                    )

    async def hybrid_search(
        self,
        collection: Collection,
        query_text: str,
        limit: int = 5,
        filter_expr: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        try:
            sparse_embedding = ef([query_text])["sparse"][
                [0]
            ]  # type: ignore[index]
            dense_embedding = (
                self.embedding_client.embeddings.create(
                    input=[query_text], model=self.embedding_model
                ).data[0].embedding
            )

            sparse_search_params = {"metric_type": "IP", "params": {}}
            dense_search_params = {"metric_type": "COSINE", "params": {}}
            sparse_request = AnnSearchRequest(
                data=[sparse_embedding],
                anns_field="sparse_vector",
                param=sparse_search_params,
                limit=limit,
                expr=filter_expr,
            )
            dense_request = AnnSearchRequest(
                data=[dense_embedding],
                anns_field="dense_vector",
                param=dense_search_params,
                limit=limit,
                expr=filter_expr,
            )
            results = collection.hybrid_search(
                [sparse_request, dense_request],
                rerank=RRFRanker(60),
                limit=limit,
                output_fields=self.output_fields,
            )[0]

            for result in results:
                result["distance"] = result["distance"] / (2 / (60 + 1))

            return results
        except Exception as e:
            raise ValueError(f"Hybrid search failed: {str(e)}")

    async def rerank_results(
        self, results: List[Dict[str, Any]], query_text: str, top_n: int
    ) -> List[Dict[str, Any]]:
        try:
            if not self.reranker_base_url or not self.reranker_model:
                return results

            documents: List[str] = []
            for item in results:
                entity = item.get("entity") or {}
                chunk_text = entity.get("chunk") if isinstance(entity, dict) else None
                if not chunk_text:
                    chunk_text = item.get("chunk")
                documents.append(str(chunk_text) if chunk_text is not None else "")

            url = f"{self.reranker_base_url}/rerank"
            payload = {
                "model": self.reranker_model,
                "query": query_text,
                "documents": documents,
            }

            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            api_results = data.get("results") or []
            if not isinstance(api_results, list) or not api_results:
                return results

            sorted_rerank = sorted(
                api_results, key=lambda x: x.get("relevance_score", 0.0), reverse=True
            )
            top_n = max(1, min(int(top_n), len(sorted_rerank)))
            top_rerank = sorted_rerank[:top_n]

            selected: List[Dict[str, Any]] = []
            for r in top_rerank:
                idx = r.get("index")
                score = r.get("relevance_score")
                if isinstance(idx, int) and 0 <= idx < len(results):
                    item = results[idx]
                    try:
                        item["distance"] = (
                            float(score) if score is not None else item.get("distance")
                        )
                    except Exception:
                        pass
                    selected.append(item)

            selected_sorted = sorted(
                selected, key=lambda it: (it.get("distance") or 0.0), reverse=True
            )
            return selected_sorted
        except Exception:
            return results


def _build_connector_from_env() -> MilvusConnector:
    uri = os.environ.get("MILVUS_URI")
    token = os.environ.get("MILVUS_TOKEN")
    db_name = os.environ.get("MILVUS_DB", "default")
    collection = os.environ.get("MILVUS_COLLECTION")
    embedding_base_url = os.environ.get("EMBEDDING_BASE_URL")
    embedding_model = os.environ.get("EMBEDDING_MODEL")
    reranker_base_url = os.environ.get("RERANKER_BASE_URL")
    reranker_model = os.environ.get("RERANKER_MODEL")

    missing = [
        name
        for name, val in [
            ("MILVUS_URI", uri),
            ("MILVUS_COLLECTION", collection),
            ("EMBEDDING_BASE_URL", embedding_base_url),
            ("EMBEDDING_MODEL", embedding_model),
        ]
        if not val
    ]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )

    return MilvusConnector(
        uri=uri,  # type: ignore[arg-type]
        token=token,
        db_name=db_name,
        law_collection_name=collection,  # type: ignore[arg-type]
        embedding_base_url=embedding_base_url,  # type: ignore[arg-type]
        embedding_model=embedding_model,  # type: ignore[arg-type]
        reranker_base_url=reranker_base_url,
        reranker_model=reranker_model,
    )


async def law_hybrid_search(
    query_text: str,
    limit: int = 50,
    filter_expr: Optional[str] = None,
    rerank: bool = True,
    rerank_limit: int = 15,
) -> str:
    logger.info(
        "Calling law_hybrid_search with params: %s",
        _preview(
            {
                "query_text": query_text,
                "limit": limit,
                "filter_expr": filter_expr,
                "rerank": rerank,
                "rerank_limit": rerank_limit,
            }
        ),
    )

    connector = _build_connector_from_env()

    results = await connector.hybrid_search(
        collection=connector.law_collection,
        query_text=query_text,
        limit=limit,
        filter_expr=filter_expr,
    )
    if not results:
        logger.info(
            "First search returned empty in law_hybrid_search, retrying without filter_expr"
        )
        try:
            results = await connector.hybrid_search(
                collection=connector.law_collection,
                query_text=query_text,
                limit=limit,
                filter_expr=None,
            )
        except Exception as e:
            logger.warning("Retry without filter_expr failed in law_hybrid_search: %s", e)

    if rerank and results:
        try:
            results = await connector.rerank_results(
                results, query_text, min(rerank_limit, limit)
            )
        except Exception as e:
            logger.warning("Rerank failed in law_hybrid_search: %s", e)

    response = format_grouped_sources(results)
    logger.info("Finished law_hybrid_search, response preview: %s", _preview(response, 1000))
    return response


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone law_hybrid_search runner. Uses env MILVUS_*, EMBEDDING_*, RERANKER_*"
        )
    )
    parser.add_argument("--query", "-q", required=True, help="Query text")
    parser.add_argument("--limit", type=int, default=50, help="Max results to return")
    parser.add_argument(
        "--filter",
        dest="filter_expr",
        default=None,
        help="Optional filter expression, e.g. chapter like \"%第三章%\"",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable external reranker (default enabled)",
    )
    parser.add_argument(
        "--rerank-limit",
        type=int,
        default=15,
        help="Maximum number of results to rerank and return",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = _parse_args()
    rerank_enabled = not args.no_rerank
    # 运行异步函数
    import asyncio

    response = asyncio.run(
        law_hybrid_search(
            query_text=args.query,
            limit=args.limit,
            filter_expr=args.filter_expr,
            rerank=rerank_enabled,
            rerank_limit=args.rerank_limit,
        )
    )
    print(response)


if __name__ == "__main__":
    main()
