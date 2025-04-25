#!/usr/bin/env python3
"""
Script to process text files, chunk them, create embeddings, and store them in Milvus.
This script implements the full deployment workflow for inserting data into a Milvus collection.

Environment variables:
- COLLECTION_NAME: Name of the Milvus collection to create/use
- MILVUS_URI: Milvus connection URI
- EMBEDDING_BASE_URL: Base URL for the embedding model API
- EMBEDDING_MODEL: Name of the embedding model to use
- EMBEDDING_DIM: Dimension of the embedding vectors
"""

import os
import glob
from typing import List, Dict, Any
import logging

from tqdm import tqdm
from openai import OpenAI
from pymilvus import MilvusClient, DataType

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
COLLECTION_NAME = os.environ.get("COLLECTION_NAME")
MILVUS_URI = os.environ.get("MILVUS_URI")
MILVUS_TOKEN = os.environ.get("MILVUS_TOKEN")
EMBEDDING_BASE_URL = os.environ.get("EMBEDDING_BASE_URL")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM"))

# Constants
CHUNK_SIZE = 512  # Token size for each chunk
CHUNK_OVERLAP = 64  # Token overlap between chunks


def load_documents() -> Dict[str, str]:
    """Load document files from workspace/files directory.
    
    Returns:
        Dict mapping filenames to file contents
    """
    file_contents = {}

    # Process all files (full deployment)
    logger.info("Processing all files (full deployment)...")
    file_pattern = ["/workspace/files/**/*.txt", "/workspace/files/**/*.md"]
    files_to_process = []
    for pattern in file_pattern:
        files_to_process.extend(glob.glob(pattern, recursive=True))

    logger.info(f"Found {len(files_to_process)} files to process")

    # Load file contents
    for file_path in files_to_process:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                if content:
                    relative_path = os.path.relpath(file_path,
                                                    "/workspace/files/")
                    file_contents[relative_path] = content
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")

    return file_contents


def chunk_text(text: str, filename: str) -> List[Dict[str, Any]]:
    """Split text into chunks with metadata.
    
    Args:
        text: Text content to chunk
        filename: Source filename for metadata
        
    Returns:
        List of chunk objects with text and metadata
    """
    # Simple chunking by paragraphs first, then by size
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # If adding this paragraph would exceed chunk size, save current chunk and start a new one
        if len(current_chunk) + len(paragraph) > CHUNK_SIZE:
            if current_chunk:
                chunks.append({
                    "text": current_chunk,
                    "metadata": {
                        "source": filename
                    }
                })
            current_chunk = paragraph
        else:
            # Add paragraph to current chunk with a separator if needed
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph

    # Add the final chunk if there's any content left
    if current_chunk:
        chunks.append({
            "text": current_chunk,
            "metadata": {
                "source": filename
            }
        })

    return chunks


def create_embedding(text: str, client: OpenAI) -> List[float]:
    """Create embedding for text using OpenAI API.
    
    Args:
        text: Input text to embed
        client: OpenAI client instance
        
    Returns:
        Embedding vector
    """
    try:
        response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error creating embedding: {e}")
        # Return a zero vector as fallback
        return [0.0] * EMBEDDING_DIM


def setup_milvus(client: MilvusClient) -> None:
    """Setup Milvus collection.
    
    Args:
        client: Milvus client instance
    """
    # Check if collection exists
    if client.has_collection(COLLECTION_NAME):
        logger.error(f"Collection {COLLECTION_NAME} already exists")
        raise RuntimeError(
            f"Collection {COLLECTION_NAME} already exists. Please drop the collection or change the collection name manually."
        )

    # Create schema using the pattern from full-text-search.py
    schema = MilvusClient.create_schema()
    schema.add_field(field_name="id",
                     datatype=DataType.INT64,
                     is_primary=True,
                     auto_id=True)
    schema.add_field(field_name="text",
                     datatype=DataType.VARCHAR,
                     max_length=65535)
    schema.add_field(field_name="source",
                     datatype=DataType.VARCHAR,
                     max_length=255)
    schema.add_field(field_name="vector",
                     datatype=DataType.FLOAT_VECTOR,
                     dim=EMBEDDING_DIM)

    # Set up index parameters
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="IP"  # Inner product distance
    )

    logger.info(f"Creating collection {COLLECTION_NAME}")
    client.create_collection(collection_name=COLLECTION_NAME,
                             schema=schema,
                             index_params=index_params)


def process_documents(documents: Dict[str, str], milvus_client: MilvusClient,
                      embedding_client: OpenAI) -> None:
    """Process documents, create chunks, and store in Milvus.
    
    Args:
        documents: Dictionary of documents {filename: content}
        milvus_client: Milvus client instance
        embedding_client: OpenAI client for embeddings
    """
    all_chunks = []

    logger.info(f"Processing {len(documents)} documents")
    for filename, content in documents.items():
        logger.info(f"Processing document: {filename}")
        chunks = chunk_text(content, filename)
        all_chunks.extend(chunks)

    logger.info(f"Created {len(all_chunks)} chunks")

    # Create embeddings and prepare data for Milvus
    milvus_data = []
    for i, chunk in enumerate(tqdm(all_chunks, desc="Creating embeddings")):
        embedding = create_embedding(chunk["text"], embedding_client)
        milvus_data.append({
            "text": chunk["text"],
            "source": chunk["metadata"]["source"],
            "vector": embedding
        })

    # Insert data into Milvus
    if milvus_data:
        logger.info(
            f"Inserting {len(milvus_data)} documents into Milvus collection {COLLECTION_NAME}"
        )
        milvus_client.insert(COLLECTION_NAME, milvus_data)
        logger.info("Data insertion complete")
    else:
        logger.warning("No data to insert")


def main():
    """Main function to run the data processing and insertion workflow."""
    logger.info("Starting insert-data workflow")

    # Initialize clients
    embedding_client = OpenAI(base_url=EMBEDDING_BASE_URL, api_key="dummy")
    milvus_client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)

    # Setup Milvus collection
    setup_milvus(milvus_client)

    # Load and process documents
    documents = load_documents()
    process_documents(documents, milvus_client, embedding_client)

    logger.info("Workflow completed successfully")


if __name__ == "__main__":
    main()
