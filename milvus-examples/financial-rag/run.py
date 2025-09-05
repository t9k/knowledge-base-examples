"""A RAG (Retrieval-Augmented Generation) implementation using Milvus and OpenAI.

This script implements a simple RAG system that:
1. Loads markdown documents
2. Creates embeddings using OpenAI
3. Stores them in Milvus
4. Performs similarity search
5. Generates answers using OpenAI
"""

from glob import glob
import json
from typing import List, Tuple

from tqdm import tqdm
from openai import OpenAI
from pymilvus import MilvusClient

# Constants
MILVUS_URI = "http://app-milvus-1-bf735a18.dev-xyx.svc.cluster.local:19530"
COLLECTION_NAME = "amazon_reviews_2023"

CHAT_BASE_URL = "http://app-vllm-d72b6b24.dev-xyx.nc201.ksvc.tensorstack.net/v1"
CHAT_MODEL = "Qwen2.5-7B-Instruct"
EMBEDDING_BASE_URL = "http://app-vllm-ba146b24.dev-xyx.nc201.ksvc.tensorstack.net/v1"
EMBEDDING_MODEL = "bge-m3"
EMBEDDING_DIM = 1024

SYSTEM_PROMPT = """You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided."""


def load_documents(docs_path: str) -> List[str]:
    """Load and split markdown documents into sections.

    Args:
        docs_path: Path pattern to markdown files

    Returns:
        List of text sections split by '#' headers
    """
    text_lines = []
    for file_path in glob(docs_path, recursive=True):
        with open(file_path, "r") as file:
            file_text = file.read()
        text_lines.extend(file_text.split("# "))
    return text_lines


def create_embedding(text: str, client: OpenAI) -> List[float]:
    """Create embedding for text using OpenAI API.

    Args:
        text: Input text to embed
        client: OpenAI client instance

    Returns:
        Embedding vector
    """
    return client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    ).data[0].embedding


def setup_milvus(client: MilvusClient) -> None:
    """Setup Milvus collection.

    Args:
        client: Milvus client instance
    """
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=EMBEDDING_DIM,
        metric_type="IP",  # Inner product distance
        consistency_level="Strong",
    )

def store_embeddings(
    texts: List[str],
    milvus_client: MilvusClient,
    openai_client: OpenAI
) -> None:
    """Create and store embeddings in Milvus.

    Args:
        texts: List of text sections to embed
        milvus_client: Milvus client instance
        openai_client: OpenAI client instance
    """
    data = []
    for i, text in enumerate(tqdm(texts, desc="Creating embeddings")):
        data.append({
            "id": i,
            "vector": create_embedding(text, openai_client),
            "text": text
        })
    milvus_client.insert(collection_name=COLLECTION_NAME, data=data)


def search_similar_texts(
    question: str,
    milvus_client: MilvusClient,
    openai_client: OpenAI,
    limit: int = 3
) -> List[Tuple[str, float]]:
    """Search for similar texts in Milvus.

    Args:
        question: Query text
        milvus_client: Milvus client instance
        openai_client: OpenAI client instance
        limit: Number of results to return

    Returns:
        List of (text, similarity_score) tuples
    """
    search_res = milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=[create_embedding(question, openai_client)],
        limit=limit,
        search_params={"metric_type": "IP", "params": {}},
        output_fields=["text"],
    )

    return [(res["entity"]["text"], res["distance"]) for res in search_res[0]]


def generate_answer(
    question: str,
    context: str,
    client: OpenAI
) -> str:
    """Generate answer using OpenAI.

    Args:
        question: User question
        context: Retrieved context
        client: OpenAI client instance

    Returns:
        Generated answer
    """
    user_prompt = f"""
    Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
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
    """Main function to run the RAG pipeline."""
    # Initialize clients
    emb_client = OpenAI(base_url=EMBEDDING_BASE_URL, api_key="dummy")
    chat_client = OpenAI(base_url=CHAT_BASE_URL, api_key="dummy")
    milvus_client = MilvusClient(uri=MILVUS_URI)

    # Load and process documents
    texts = load_documents("milvus_docs/en/faq/*.md")
    
    # Setup and populate Milvus
    setup_milvus(milvus_client)
    store_embeddings(texts, milvus_client, emb_client)


    # Example question
    question = "How is data stored in milvus?"
    
    # Search and retrieve similar texts
    retrieved_texts = search_similar_texts(
        question,
        milvus_client,
        emb_client
    )

    # Print search results
    print("Search Results:")
    print(json.dumps(retrieved_texts, indent=4))

    # Generate and print answer
    context = "\n".join(text for text, _ in retrieved_texts)
    answer = generate_answer(question, context, chat_client)
    print("\nGenerated Answer:")
    print(answer)


if __name__ == "__main__":
    main()
