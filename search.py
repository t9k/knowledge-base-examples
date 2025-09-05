import time

from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus.model.reranker import BGERerankFunction
from pymilvus import (
    connections,
    Collection,
    AnnSearchRequest,
    RRFRanker,
)
from openai import OpenAI
import argparse

# MILVUS_URI = "http://app-milvus-a4b42611.demo.svc.cluster.local:19530"
# COLLECTION_NAME = "criminal_cases"
MILVUS_URI = "http://app-milvus-6f97-24.demo.svc.cluster.local:19530"
COLLECTION_NAME = "criminal_law_cases"

CHAT_BASE_URL = "http://app-vllm-enflame-8c45424c.demo.ksvc.qy.t9kcloud.cn/v1"
CHAT_MODEL = "Qwen2.5-72B-Instruct"
SYSTEM_PROMPT = """你是一名法律智能助手，擅长根据提供的刑事案件案情描述和事实上下文，准确、简明地回答用户提出的刑事法律问题。你的回答应基于上下文信息，条理清晰、专业可靠，避免主观臆断，不可凭空编造。"""
TOP_K = 10


def perform_search(col,
                   filter,
                   query_dense_embedding,
                   query_sparse_embedding,
                   limit,
                   search_mode="hybrid"):
    output_fields = [
        "chunk", "relevant_articles", "accusation", "punish_of_money",
        "criminal", "imprisonment", "life_imprisonment", "death_penalty",
        "dates", "amounts", "victim", "defendant"
    ]

    if search_mode == "hybrid":
        # 混合搜索
        dense_search_params = {"metric_type": "COSINE", "params": {}}
        dense_req = AnnSearchRequest([query_dense_embedding],
                                     "dense_vector",
                                     dense_search_params,
                                     limit=limit,
                                     expr=filter)
        sparse_search_params = {"metric_type": "IP", "params": {}}
        sparse_req = AnnSearchRequest([query_sparse_embedding],
                                      "sparse_vector",
                                      sparse_search_params,
                                      limit=limit,
                                      expr=filter)
        rerank = RRFRanker(60)
        res = col.hybrid_search([sparse_req, dense_req],
                                rerank=rerank,
                                limit=limit,
                                output_fields=output_fields)[0]
    elif search_mode == "dense":
        # 密集向量搜索
        dense_search_params = {"metric_type": "COSINE", "params": {}}
        res = col.search(data=[query_dense_embedding],
                         anns_field="dense_vector",
                         param=dense_search_params,
                         limit=limit,
                         filter=filter,
                         output_fields=output_fields)[0]
    elif search_mode == "sparse":
        # 稀疏向量搜索
        sparse_search_params = {"metric_type": "IP", "params": {}}
        res = col.search(data=[query_sparse_embedding],
                         anns_field="sparse_vector",
                         param=sparse_search_params,
                         limit=limit,
                         filter=filter,
                         output_fields=output_fields)[0]
    else:
        raise ValueError(f"Unsupported search mode: {search_mode}")

    return res


def generate_answer(question: str, context: str, client: OpenAI) -> str:
    user_prompt = f"""
请你根据<context>标签内给出的刑事案件事实、相关法条等上下文信息，认真、准确地回答<question>标签内的问题。
要求：
1. 回答必须严格依据上下文信息，不得凭空编造。
2. 回答应简明扼要、条理清晰。
3. 如上下文无法直接回答问题，请明确说明"根据已知信息无法作答"。

<context>
{context}
</context>
<question>
{question}
</question>
    """

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            },
        ],
    )
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="用户问题")
    parser.add_argument("--rag",
                        action="store_true",
                        help="是否执行RAG生成（若不加此参数则只做检索）")
    parser.add_argument("--search_mode",
                        type=str,
                        default="hybrid",
                        choices=["dense", "sparse", "hybrid"],
                        help="检索模式: dense(密集向量), sparse(稀疏向量), hybrid(混合检索)")
    parser.add_argument("--filter", type=str, default=None, help="过滤条件")
    parser.add_argument("--use_gcu",
                        action="store_true",
                        help="是否使用Enflame GCU进行embedding")
    parser.add_argument("--use_llm_reranker",
                        action="store_true",
                        help="是否使用LLM reranker（仅在hybrid模式下生效）")
    args = parser.parse_args()

    query = args.query
    do_rag = args.rag
    search_mode = args.search_mode
    use_llm_reranker = args.use_llm_reranker
    print(f"Query: {query}")
    print(f"Search Mode: {search_mode}")
    if use_llm_reranker and search_mode == "hybrid":
        print("Using LLM reranker for hybrid search")

    connections.connect(uri=MILVUS_URI, token="root:Milvus")
    col = Collection(COLLECTION_NAME)

    time_start = time.time()
    if args.use_gcu:
        import torch
        import torch_gcu
        ef = BGEM3EmbeddingFunction(use_fp16=False, device="gcu")
        rf = BGERerankFunction(device="gcu")
        print("Using Enflame GCU for embedding")
    else:
        ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
        rf = BGERerankFunction(device="cpu")
        print("Using CPU for embedding")
    
    query_embeddings = ef([query])
    results = perform_search(
        col,
        args.filter,
        query_embeddings["dense"][0],
        query_embeddings["sparse"][[0]],
        limit=TOP_K,
        search_mode=search_mode
    )
    
    time_end = time.time()

    if use_llm_reranker and search_mode == "hybrid":
        chunks = [hit.chunk for hit in results]
        rank_results = rf(query, chunks, TOP_K)
        
        new_results = []
        for r in rank_results:
            new_result = results[r.index]
            new_result["score"] = r.score
            new_results.append(new_result)

        results = new_results
    
    chunks = []
    print("\nTop Results:")
    for hit in results:
        if search_mode == "hybrid":
            if use_llm_reranker:
                score = hit["score"]
            else:
                score = hit.score / (2 / (60 + 1))
        else:
            score = hit.score

        chunks.append(hit.chunk)
        print(f"score: {score}")
        print(f"chunk: {hit.chunk}")
        print(f"  relevant_articles: {hit.relevant_articles}")
        print(f"  accusation: {hit.accusation}")
        print(f"  punish_of_money: {hit.punish_of_money}")
        print(f"  criminal: {hit.criminal}")
        print(f"  imprisonment: {hit.imprisonment}")
        print(f"  life_imprisonment: {hit.life_imprisonment}")
        print(f"  death_penalty: {hit.death_penalty}")
        print("-")

    if do_rag:
        chat_client = OpenAI(base_url=CHAT_BASE_URL, api_key="dummy")
        context = "\n\n".join(chunks)
        answer = generate_answer(query, context, chat_client)
        print("\nGenerated Answer:")
        print(answer)


if __name__ == "__main__":
    main()
