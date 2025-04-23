"""A hybrid search implementation using Milvus vector database.

This script implements a hybrid search system that:
1. Creates a Milvus collection with both sparse and dense vector fields
2. Loads document chunks and inserts them into Milvus
3. Supports three search modes: sparse (BM25), dense (vector), and hybrid
4. Evaluates retrieval performance using Pass@k metric
"""

import json

from pymilvus import (
    MilvusClient,
    DataType,
    Function,
    FunctionType,
    AnnSearchRequest,
    RRFRanker,
)
from pymilvus.model.dense import OpenAIEmbeddingFunction


# Constants
MILVUS_URI = "http://app-milvus-xxxxxxxx.namespace.svc.cluster.local:19530"
COLLECTION_NAME = "milvus_hybrid"

EMBEDDING_BASE_URL = "http://app-vllm-xxxxxxxx.namespace.ksvc.tensorstack.net/v1"
EMBEDDING_MODEL = "bge-m3"

TOPK = 5


class HybridRetriever:
    def __init__(self, uri, collection_name="hybrid", dense_embedding_function=None):
        """Initialize a hybrid retriever that supports sparse, dense and hybrid search.
        
        Args:
            uri: Milvus server URI
            collection_name: Name of the Milvus collection
            dense_embedding_function: Function to create dense embeddings
        """
        self.uri = uri
        self.collection_name = collection_name
        self.embedding_function = dense_embedding_function
        self.use_reranker = True
        self.use_sparse = True
        self.client = MilvusClient(uri=uri)

    def build_collection(self):
        """Create a Milvus collection with schema for hybrid search."""
        dense_dim = 4096

        tokenizer_params = {
            "tokenizer": "standard",
            "filter": [
                "lowercase",
                {
                    "type": "length",
                    "max": 200,
                },
                {"type": "stemmer", "language": "english"},
                {
                    "type": "stop",
                    "stop_words": [
                        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", 
                        "if", "in", "into", "is", "it", "no", "not", "of", "on", "or", 
                        "such", "that", "the", "their", "then", "there", "these", "they", 
                        "this", "to", "was", "will", "with",
                    ],
                },
            ],
        }

        schema = MilvusClient.create_schema()
        schema.add_field(
            field_name="pk",
            datatype=DataType.VARCHAR,
            is_primary=True,
            auto_id=True,
            max_length=100,
        )
        schema.add_field(
            field_name="content",
            datatype=DataType.VARCHAR,
            max_length=65535,
            analyzer_params=tokenizer_params,
            enable_match=True,
            enable_analyzer=True,
        )
        schema.add_field(
            field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR
        )
        schema.add_field(
            field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dense_dim
        )
        schema.add_field(
            field_name="original_uuid", datatype=DataType.VARCHAR, max_length=128
        )
        schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(
            field_name="chunk_id", datatype=DataType.VARCHAR, max_length=64
        ),
        schema.add_field(field_name="original_index", datatype=DataType.INT32)

        functions = Function(
            name="bm25",
            function_type=FunctionType.BM25,
            input_field_names=["content"],
            output_field_names="sparse_vector",
        )

        schema.add_function(functions)

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )
        index_params.add_index(
            field_name="dense_vector", index_type="FLAT", metric_type="IP"
        )

        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def insert_data(self, chunk, metadata):
        """Insert a document chunk with its metadata into Milvus.
        
        Args:
            chunk: Document chunk text content
            metadata: Metadata about the chunk
        """
        embedding = self.embedding_function([chunk])
        if isinstance(embedding, dict) and "dense" in embedding:
            dense_vec = embedding["dense"][0]
        else:
            dense_vec = embedding[0]
        self.client.insert(
            self.collection_name, {"dense_vector": dense_vec, **metadata}
        )

    def search(self, query: str, k: int = 20, mode="hybrid"):
        """Search for relevant documents using sparse, dense, or hybrid search.
        
        Args:
            query: Search query text
            k: Number of results to return
            mode: Search mode ('sparse', 'dense', or 'hybrid')
            
        Returns:
            List of document chunks with scores
        """
        output_fields = [
            "content",
            "original_uuid",
            "doc_id",
            "chunk_id",
            "original_index",
        ]
        if mode in ["dense", "hybrid"]:
            embedding = self.embedding_function([query])
            if isinstance(embedding, dict) and "dense" in embedding:
                dense_vec = embedding["dense"][0]
            else:
                dense_vec = embedding[0]

        if mode == "sparse":
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query],
                anns_field="sparse_vector",
                limit=k,
                output_fields=output_fields,
            )
        elif mode == "dense":
            results = self.client.search(
                collection_name=self.collection_name,
                data=[dense_vec],
                anns_field="dense_vector",
                limit=k,
                output_fields=output_fields,
            )
        elif mode == "hybrid":
            full_text_search_params = {"metric_type": "BM25"}
            full_text_search_req = AnnSearchRequest(
                [query], "sparse_vector", full_text_search_params, limit=k
            )

            dense_search_params = {"metric_type": "IP"}
            dense_req = AnnSearchRequest(
                [dense_vec], "dense_vector", dense_search_params, limit=k
            )

            results = self.client.hybrid_search(
                self.collection_name,
                [full_text_search_req, dense_req],
                ranker=RRFRanker(),
                limit=k,
                output_fields=output_fields,
            )
        else:
            raise ValueError("Invalid mode")
        return [
            {
                "doc_id": doc["entity"]["doc_id"],
                "chunk_id": doc["entity"]["chunk_id"],
                "content": doc["entity"]["content"],
                "score": doc["distance"],
            }
            for doc in results[0]
        ]


def load_jsonl(file_path: str):
    """Load JSONL file and return a list of dictionaries.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of dictionaries parsed from the JSONL file
    """
    with open(file_path, "r") as file:
        return [json.loads(line) for line in file]


def load_data(path: str):
    """Load JSON data from file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Data loaded from the JSON file
    """
    with open(path, "r") as f:
        return json.load(f)


def insert_dataset(retriever, dataset):
    """Insert dataset into Milvus collection.
    
    Args:
        retriever: HybridRetriever instance
        dataset: Dataset to insert
    """
    retriever.build_collection()
    for doc in dataset:
        for chunk in doc["chunks"]:
            metadata = {
                "doc_id": doc["doc_id"],
                "original_uuid": doc["original_uuid"],
                "chunk_id": chunk["chunk_id"],
                "original_index": chunk["original_index"],
                "content": chunk["content"],
            }
            chunk_content = chunk["content"]
            retriever.insert_data(chunk_content, metadata)
    print(f"Data inserted into collection: {retriever.collection_name}")


def evaluate_retrieval(retriever, eval_dataset, mode="hybrid", k=5):
    """Evaluate the retrieval performance using Pass@k metric.
    
    Args:
        retriever: HybridRetriever instance
        eval_dataset: Evaluation dataset
        mode: Search mode to evaluate ('sparse', 'dense', or 'hybrid')
        k: Number of top results to consider
        
    Returns:
        Pass@k score (average retrieval accuracy)
    """
    total_query_score = 0
    num_queries = 0

    for query_item in eval_dataset:
        query = query_item["query"]
        golden_chunk_uuids = query_item["golden_chunk_uuids"]

        chunks_found = 0
        golden_contents = []
        for doc_uuid, chunk_index in golden_chunk_uuids:
            golden_doc = next(
                (doc for doc in query_item["golden_documents"] if doc["uuid"] == doc_uuid),
                None,
            )
            if golden_doc:
                golden_chunk = next(
                    (
                        chunk
                        for chunk in golden_doc["chunks"]
                        if chunk["index"] == chunk_index
                    ),
                    None,
                )
                if golden_chunk:
                    golden_contents.append(golden_chunk["content"].strip())

        results = retriever.search(query, mode=mode, k=k)

        for golden_content in golden_contents:
            for doc in results[:k]:
                retrieved_content = doc["content"].strip()
                if retrieved_content == golden_content:
                    chunks_found += 1
                    break

        query_score = chunks_found / len(golden_contents) if golden_contents else 0
        total_query_score += query_score
        num_queries += 1

    return total_query_score / num_queries if num_queries > 0 else 0


def main():
    """Main function to run the full text search pipeline."""
    # Initialize embedding function and retriever
    dense_ef = OpenAIEmbeddingFunction(base_url=EMBEDDING_BASE_URL, api_key="dummy", model_name=EMBEDDING_MODEL)
    retriever = HybridRetriever(
        uri=MILVUS_URI,
        collection_name=COLLECTION_NAME,
        dense_embedding_function=dense_ef,
    )

    # Step 1: Insert dataset
    print("Step 1: Inserting data into Milvus...")
    dataset = load_data("codebase_chunks.json")
    insert_dataset(retriever, dataset)
    
    # Step 2: Test sparse search with a sample query
    print("\nStep 2: Testing sparse search...")
    sample_query = "create a logger?"
    sparse_results = retriever.search(sample_query, mode="sparse", k=3)
    print(f"Query: {sample_query}")
    print(f"Results (sparse search):")
    for i, result in enumerate(sparse_results):
        print(f"{i+1}. {result['doc_id']} ({result['score']:.4f})")
        print(f"   {result['content'][:100]}...")

    # Step 3: Run evaluation with hybrid search
    print("\nStep 3: Running evaluation with hybrid search...")
    eval_dataset = load_jsonl("evaluation_set.jsonl")
    pass_at_k = evaluate_retrieval(retriever, eval_dataset, mode="hybrid", k=TOPK)
    print(f"Pass@{TOPK}: {pass_at_k:.4f}")


if __name__ == "__main__":
    main()
