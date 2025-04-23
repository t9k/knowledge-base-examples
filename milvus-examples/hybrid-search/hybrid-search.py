"""
Hybrid Search with Dense and Sparse Vectors in Milvus

This script demonstrates how to conduct hybrid search with Milvus and BGE-M3 model.
BGE-M3 model can convert text into dense and sparse vectors. Milvus supports storing
both types of vectors in one collection, allowing for hybrid search that enhances
the result relevance.
"""

import pandas as pd
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    AnnSearchRequest,
    WeightedRanker,
)

# Constants
MILVUS_URI = "http://app-milvus-xxxxxxxx.namespace.svc.cluster.local:19530"
COLLECTION_NAME = "hybrid_demo"


def load_data(max_docs=500):
    """
    Load and prepare data from the Quora Duplicate Questions dataset.
    
    Args:
        max_docs: Maximum number of documents to load
        
    Returns:
        List of questions
    """
    file_path = "quora_duplicate_questions.tsv"
    df = pd.read_csv(file_path, sep="\t")
    questions = set()
    for _, row in df.iterrows():
        obj = row.to_dict()
        questions.add(obj["question1"][:512])
        questions.add(obj["question2"][:512])
        if len(questions) > max_docs:
            break
    
    return list(questions)

def setup_milvus_collection(dense_dim):
    """
    Set up the Milvus collection and create indices for the vector fields.
    
    Args:
        dense_dim: Dimension of the dense vector
        
    Returns:
        Milvus collection object
    """
    # Connect to Milvus given URI
    connections.connect(uri=MILVUS_URI)
    
    # Specify the data schema for the new Collection
    fields = [
        # Use auto generated id as primary key
        FieldSchema(
            name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100
        ),
        # Store the original text to retrieve based on semantically distance
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
        # Milvus now supports both sparse and dense vectors,
        # we can store each in a separate field to conduct hybrid search on both vectors
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
    ]
    schema = CollectionSchema(fields)
    
    # Create collection (drop the old one if exists)
    col_name = COLLECTION_NAME
    if utility.has_collection(col_name):
        Collection(col_name).drop()
    col = Collection(col_name, schema, consistency_level="Strong")
    
    # To make vector search efficient, we need to create indices for the vector fields
    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
    col.create_index("sparse_vector", sparse_index)
    dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
    col.create_index("dense_vector", dense_index)
    col.load()
    
    return col

def insert_data(col, docs, docs_embeddings):
    """
    Insert documents and their embeddings into the collection.
    
    Args:
        col: Milvus collection object
        docs: List of documents
        docs_embeddings: Embeddings of the documents
        
    Returns:
        Number of entities inserted
    """
    # For efficiency, we insert 50 records in each small batch
    for i in range(0, len(docs), 50):
        batched_entities = [
            docs[i : i + 50],
            docs_embeddings["sparse"][i : i + 50],
            docs_embeddings["dense"][i : i + 50],
        ]
        col.insert(batched_entities)
    
    return col.num_entities

def dense_search(col, query_dense_embedding, limit=10):
    """
    Search only across dense vector field.
    
    Args:
        col: Milvus collection object
        query_dense_embedding: Dense embedding of the query
        limit: Maximum number of results to return
        
    Returns:
        List of search results
    """
    search_params = {"metric_type": "IP", "params": {}}
    res = col.search(
        [query_dense_embedding],
        anns_field="dense_vector",
        limit=limit,
        output_fields=["text"],
        param=search_params,
    )[0]
    return [hit.get("text") for hit in res]

def sparse_search(col, query_sparse_embedding, limit=10):
    """
    Search only across sparse vector field.
    
    Args:
        col: Milvus collection object
        query_sparse_embedding: Sparse embedding of the query
        limit: Maximum number of results to return
        
    Returns:
        List of search results
    """
    search_params = {
        "metric_type": "IP",
        "params": {},
    }
    res = col.search(
        [query_sparse_embedding],
        anns_field="sparse_vector",
        limit=limit,
        output_fields=["text"],
        param=search_params,
    )[0]
    return [hit.get("text") for hit in res]

def hybrid_search(
    col,
    query_dense_embedding,
    query_sparse_embedding,
    sparse_weight=1.0,
    dense_weight=1.0,
    limit=10,
):
    """
    Search across both dense and sparse vector fields with a weighted reranker.
    
    Args:
        col: Milvus collection object
        query_dense_embedding: Dense embedding of the query
        query_sparse_embedding: Sparse embedding of the query
        sparse_weight: Weight for sparse search
        dense_weight: Weight for dense search
        limit: Maximum number of results to return
        
    Returns:
        List of search results
    """
    dense_search_params = {"metric_type": "IP", "params": {}}
    dense_req = AnnSearchRequest(
        [query_dense_embedding], "dense_vector", dense_search_params, limit=limit
    )
    sparse_search_params = {"metric_type": "IP", "params": {}}
    sparse_req = AnnSearchRequest(
        [query_sparse_embedding], "sparse_vector", sparse_search_params, limit=limit
    )
    rerank = WeightedRanker(sparse_weight, dense_weight)
    res = col.hybrid_search(
        [sparse_req, dense_req], rerank=rerank, limit=limit, output_fields=["text"]
    )[0]
    return [hit.get("text") for hit in res]

def doc_text_formatting(ef, query, docs):
    """
    Format the search results with highlighting.
    
    Args:
        ef: Embedding function object
        query: Search query
        docs: List of search results
        
    Returns:
        List of formatted search results
    """
    tokenizer = ef.model.tokenizer
    query_tokens_ids = tokenizer.encode(query, return_offsets_mapping=True)
    query_tokens = tokenizer.convert_ids_to_tokens(query_tokens_ids)
    formatted_texts = []

    for doc in docs:
        ldx = 0
        landmarks = []
        encoding = tokenizer.encode_plus(doc, return_offsets_mapping=True)
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])[1:-1]
        offsets = encoding["offset_mapping"][1:-1]
        for token, (start, end) in zip(tokens, offsets):
            if token in query_tokens:
                if len(landmarks) != 0 and start == landmarks[-1]:
                    landmarks[-1] = end
                else:
                    landmarks.append(start)
                    landmarks.append(end)
        close = False
        formatted_text = ""
        for i, c in enumerate(doc):
            if ldx == len(landmarks):
                pass
            elif i == landmarks[ldx]:
                if close:
                    formatted_text += "</span>"
                else:
                    formatted_text += "<span style='color:red'>"
                close = not close
                ldx = ldx + 1
            formatted_text += c
        if close is True:
            formatted_text += "</span>"
        formatted_texts.append(formatted_text)
    return formatted_texts

def display_results(query, dense_results, sparse_results, hybrid_results, ef):
    """
    Display search results with highlights.
    
    Args:
        query: Search query
        dense_results: Results from dense search
        sparse_results: Results from sparse search
        hybrid_results: Results from hybrid search
        ef: Embedding function object
    """
    print("Dense Search Results:")
    for result in dense_results:
        print(f"- {result}")
    
    print("\nSparse Search Results:")
    formatted_results = doc_text_formatting(ef, query, sparse_results)
    for i, result in enumerate(sparse_results):
        # Remove HTML formatting for console output
        formatted = formatted_results[i].replace("<span style='color:red'>", "*").replace("</span>", "*")
        print(f"- {formatted}")
    
    print("\nHybrid Search Results:")
    formatted_results = doc_text_formatting(ef, query, hybrid_results)
    for i, result in enumerate(hybrid_results):
        # Remove HTML formatting for console output
        formatted = formatted_results[i].replace("<span style='color:red'>", "*").replace("</span>", "*")
        print(f"- {formatted}")

def main():
    """Main function to run the hybrid search demo."""
    # Load and prepare data
    docs = load_data(max_docs=500)
    print(f"Loaded {len(docs)} documents")
    print(f"Example document: {docs[0]}")
    
    # Use BGE-M3 Model for Embeddings
    print("Generating embeddings using BGE-M3 model...")
    ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
    dense_dim = ef.dim["dense"]
    docs_embeddings = ef(docs)
    
    # Setup Milvus Collection and Index
    print("Setting up Milvus collection...")
    col = setup_milvus_collection(dense_dim)
    
    # Insert Data into Milvus Collection
    num_entities = insert_data(col, docs, docs_embeddings)
    print(f"Number of entities inserted: {num_entities}")
    
    # Get search query from user
    query = "How to start learning programming?"
    
    # Generate embeddings for the query
    print("Generating embeddings for the query...")
    query_embeddings = ef([query])
    
    # Run the searches
    print("Running searches...")
    dense_results = dense_search(col, query_embeddings["dense"][0])
    sparse_results = sparse_search(col, query_embeddings["sparse"][[0]])
    hybrid_results = hybrid_search(
        col,
        query_embeddings["dense"][0],
        query_embeddings["sparse"][[0]],
        sparse_weight=0.7,
        dense_weight=1.0,
    )
    
    # Display search results
    display_results(query, dense_results, sparse_results, hybrid_results, ef)

if __name__ == "__main__":
    main()
