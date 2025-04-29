import sys
import json
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus import (
    connections,
    Collection,
    AnnSearchRequest,
    RRFRanker,
)
from openai import OpenAI
import argparse

MILVUS_URI = "http://app-milvus-xxxxxxxx.namespace.svc.cluster.local:19530"
COLLECTION_NAME = "law_hybrid_demo"

CHAT_BASE_URL = "http://app-vllm-xxxxxxxx.namespace.ksvc.tensorstack.net/v1"
CHAT_MODEL = "Qwen2.5-7B-Instruct"
SYSTEM_PROMPT = """你是一名法律智能助手，擅长根据提供的刑事案件案情描述和事实上下文，准确、简明地回答用户提出的刑事法律问题。你的回答应基于上下文信息，条理清晰、专业可靠，避免主观臆断，不可凭空编造。"""


def hybrid_search(col, query_dense_embedding, query_sparse_embedding, limit=10):
    dense_search_params = {"metric_type": "IP", "params": {}}
    dense_req = AnnSearchRequest(
        [query_dense_embedding], "dense_vector", dense_search_params, limit=limit
    )
    sparse_search_params = {"metric_type": "IP", "params": {}}
    sparse_req = AnnSearchRequest(
        [query_sparse_embedding], "sparse_vector", sparse_search_params, limit=limit
    )
    rerank = RRFRanker(60)
    res = col.hybrid_search(
        [sparse_req, dense_req], rerank=rerank, limit=limit,
        output_fields=[
            "fact", "relevant_articles", "accusation", "punish_of_money", "criminals", "imprisonment", "life_imprisonment", "death_penalty"
        ]
    )[0]
    return res

def generate_answer(question: str, context: str, client: OpenAI) -> str:
    # 4. 如果上下文中出现数额"1万某"、"1.2万1某"时，应理解为"1万"，"1.2万"，"万某"或"万1某"是把万当作姓进行了错误的预处理。
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
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="用户问题")
    parser.add_argument("--rag", action="store_true", help="是否执行RAG生成（默认执行RAG，若不加此参数则只做检索）")
    args = parser.parse_args()

    query = args.query
    do_rag = args.rag
    print(f"Query: {query}")
    ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
    query_embeddings = ef([query])
    connections.connect(uri=MILVUS_URI)
    col = Collection(COLLECTION_NAME)
    results = hybrid_search(
        col,
        query_embeddings["dense"][0],
        query_embeddings["sparse"][[0]],
        limit=5,
    )
    print("\nTop Results:")
    for hit in results:
        print(f"fact: {hit.get('fact')}")
        print(f"  relevant_articles: {hit.get('relevant_articles')}")
        print(f"  accusation: {hit.get('accusation')}")
        print(f"  punish_of_money: {hit.get('punish_of_money')}")
        print(f"  criminals: {hit.get('criminals')}")
        print(f"  imprisonment: {hit.get('imprisonment')}")
        print(f"  life_imprisonment: {hit.get('life_imprisonment')}")
        print(f"  death_penalty: {hit.get('death_penalty')}")
        print("-")

    if do_rag:
        chat_client = OpenAI(base_url=CHAT_BASE_URL, api_key="dummy")
        context = "\n".join(hit.get("fact", "") for hit in results if hit.get("fact"))
        answer = generate_answer(query, context, chat_client)
        print("\nGenerated Answer:")
        print(answer)

if __name__ == "__main__":
    main()
