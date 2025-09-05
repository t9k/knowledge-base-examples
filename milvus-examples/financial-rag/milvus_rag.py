import json
from pymilvus import connections, Collection, utility
from pymilvus import CollectionSchema, FieldSchema, DataType
import numpy as np
from sentence_transformers import SentenceTransformer

def connect_to_milvus(host='localhost', port='19530'):
    """Connect to Milvus server"""
    connections.connect(host=host, port=port)
    print("Connected to Milvus server")

def create_collection(collection_name, dim=768):
    """Create a Milvus collection for document chunks"""
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    # Define fields
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="metadata", dtype=DataType.JSON)
    ]
    
    # Create collection schema
    schema = CollectionSchema(fields=fields, description="Financial document chunks")
    
    # Create collection
    collection = Collection(name=collection_name, schema=schema)
    
    # Create index for vector field
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 8, "efConstruction": 64}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    
    return collection

def load_chunks(file_path):
    """Load preprocessed chunks from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_embeddings(texts, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
    """Generate embeddings for texts using sentence-transformers"""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts)
    return embeddings

def insert_chunks_to_milvus(collection, chunks, embeddings):
    """Insert chunks and their embeddings into Milvus"""
    entities = [
        [chunk["id"] for chunk in chunks],
        [chunk["text"] for chunk in chunks],
        embeddings.tolist(),
        [chunk["metadata"] for chunk in chunks]
    ]
    
    collection.insert(entities)
    collection.flush()
    print(f"Inserted {len(chunks)} chunks into Milvus")

def search_similar_chunks(collection, query_text, model, top_k=5):
    """Search for similar chunks in Milvus"""
    # Generate embedding for query
    query_embedding = model.encode([query_text])[0]
    
    # Search
    collection.load()
    search_params = {"metric_type": "COSINE", "params": {"ef": 32}}
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["chunk_id", "text", "metadata"]
    )
    
    return results

def main():
    # Connect to Milvus
    connect_to_milvus()
    
    # Load preprocessed chunks
    chunks_file = "milvus-examples/financial-rag/processed_chunks.json"
    chunks = load_chunks(chunks_file)
    
    # Generate embeddings
    texts = [chunk["text"] for chunk in chunks]
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = generate_embeddings(texts)
    
    # Create collection and insert data
    collection_name = "byd_financial_report"
    collection = create_collection(collection_name, dim=embeddings.shape[1])
    insert_chunks_to_milvus(collection, chunks, embeddings)
    
    # Example search
    query = "比亚迪在新能源汽车市场有什么成就?"
    results = search_similar_chunks(collection, query, model)
    
    print("\nSearch results for:", query)
    for i, result in enumerate(results[0]):
        print(f"\nResult {i+1} (Score: {result.score:.4f}):")
        print(result.entity.get('text')[:200] + "...")

if __name__ == "__main__":
    # Uncomment to run the full pipeline
    # main()
    print("This script provides a complete Milvus RAG pipeline for BYD financial report")
    print("Modify the connection parameters and uncomment main() to run")