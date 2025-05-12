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
PARENT_COLLECTION_NAME = "law_hybrid_demo_parent"
CHILD_COLLECTION_NAME = "law_hybrid_demo_child"

CHAT_BASE_URL = "http://app-vllm-xxxxxxxx.namespace.ksvc.tensorstack.net/v1"
CHAT_MODEL = "Qwen2.5-7B-Instruct"
SYSTEM_PROMPT = """你是一名法律智能助手，擅长根据提供的刑事案件案情描述和事实上下文，准确、简明地回答用户提出的刑事法律问题。你的回答应基于上下文信息，条理清晰、专业可靠，避免主观臆断，不可凭空编造。"""
TOP_K = 5
RRF_K = 60
RETRIEVE_PARENT_THRESHOLD = 0.3


def hybrid_search(col, query_dense_embedding, query_sparse_embedding, limit):
    dense_search_params = {"metric_type": "COSINE", "params": {}}
    dense_req = AnnSearchRequest([query_dense_embedding],
                                 "dense_vector",
                                 dense_search_params,
                                 limit=limit)
    sparse_search_params = {"metric_type": "IP", "params": {}}
    sparse_req = AnnSearchRequest([query_sparse_embedding],
                                  "sparse_vector",
                                  sparse_search_params,
                                  limit=limit)
    rerank = RRFRanker(RRF_K)
    res = col.hybrid_search([sparse_req, dense_req],
                            rerank=rerank,
                            limit=limit,
                            output_fields=[
                                "child_id", "parent_id", "fact_id", "chunk",
                                "relevant_articles", "accusation",
                                "punish_of_money", "criminal", "imprisonment",
                                "life_imprisonment", "death_penalty"
                            ])[0]
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
    args = parser.parse_args()

    query = args.query
    do_rag = args.rag
    print(f"Query: {query}")
    ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
    query_embeddings = ef([query])
    connections.connect(uri=MILVUS_URI, token="root:Milvus")
    parent_col = Collection(PARENT_COLLECTION_NAME)
    child_col = Collection(CHILD_COLLECTION_NAME)
    child_results = hybrid_search(
        child_col,
        query_embeddings["dense"][0],
        query_embeddings["sparse"][[0]],
        limit=TOP_K,
    )

    parent_ids = [hit.parent_id for hit in child_results]
    expr = f'parent_id IN {parent_ids}'
    parent_results = parent_col.query(expr=expr, output_fields=["chunk"])
    parent_chunks = {
        parent["parent_id"]: parent["chunk"]
        for parent in parent_results
    }

    chunks = []
    print("\nTop Results:")
    for hit in child_results:

        # TODO: 根据阈值判断是否需要检索父文档
        # score = hit.score / (2 / (RRF_K + 1))
        # if score > RETRIEVE_PARENT_THRESHOLD:
        #     chunk = hit.chunk
        # else:
        #     chunk = parent_chunks.get(hit.parent_id)

        chunk = parent_chunks.get(hit.parent_id)
        chunks.append(chunk)
        print(f"chunk: {chunk}")
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
