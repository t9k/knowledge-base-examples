#!/usr/bin/env python3
"""
Script to process text files, chunk them, create embeddings, and store them in Milvus.
This script implements the full deployment workflow for inserting data into a Milvus collection.

Environment variables:
- DATABASE_NAME: Name of the Milvus database to use
- COLLECTION_NAME: Name of the Milvus collection to create
- MILVUS_URI: Milvus connection URI
- EMBEDDING_BASE_URL: Base URL for the embedding model API
- EMBEDDING_MODEL: Name of the embedding model to use
- EMBEDDING_DIM: Dimension of the embedding vectors
- LOG_FILE: Optional path to the log file
"""

import os
import glob
import sys
from typing import List, Dict, Any, Iterator, Generator, Tuple
import logging

from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI
from pymilvus import MilvusClient, DataType
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Set up logging
# Check if LOG_FILE environment variable is set
log_file = os.environ.get("LOG_FILE")
if log_file:
    # When running in the workflow with tee redirecting output to a log file,
    # only use StreamHandler to avoid duplicate logs
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])
else:
    # Default logging setup
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv("/env-config/env.properties")
DATABASE_NAME = os.environ.get("DATABASE_NAME")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME")
MILVUS_URI = os.environ.get("MILVUS_URI")
MILVUS_TOKEN = os.environ.get("MILVUS_TOKEN")
EMBEDDING_BASE_URL = os.environ.get("EMBEDDING_BASE_URL")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM"))

# Log environment variables
logger.info("=== insert-data.py environment ===")
logger.info(f"DATABASE_NAME: {DATABASE_NAME}")
logger.info(f"COLLECTION_NAME: {COLLECTION_NAME}")
logger.info(f"MILVUS_URI: {MILVUS_URI}")
logger.info(f"EMBEDDING_BASE_URL: {EMBEDDING_BASE_URL}")
logger.info(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
logger.info(f"EMBEDDING_DIM: {EMBEDDING_DIM}")
logger.info(f"LOG_FILE: {log_file}")

# Constants
CHUNK_SIZE = 512  # Token size for each chunk
CHUNK_OVERLAP = 64  # Token overlap between chunks
CHUNK_BATCH_SIZE = 100  # Batch size for text processing
FILE_BLOCK_SIZE = 1048576  # Block size for file reading


def setup_milvus(client: MilvusClient) -> None:
    """Setup Milvus collection.
    
    Args:
        client: Milvus client instance
    """
    if DATABASE_NAME not in client.list_databases():
        logger.error(f"Database {DATABASE_NAME} does not exist")
        raise RuntimeError(
            f"Database {DATABASE_NAME} does not exist. Please create the database manually."
        )

    client.using_database(DATABASE_NAME)
    logger.info(f"Using database: {DATABASE_NAME}")

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
    logger.info(f"Collection {COLLECTION_NAME} created successfully")


def create_embedding(text: str, client: OpenAI) -> List[float]:
    """Create embedding for text using OpenAI API.
    
    Args:
        text: Input text to embed
        client: OpenAI client instance
        
    Returns:
        Embedding vector
    """
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response.data[0].embedding


def create_embeddings(texts: List[Dict[str, Any]],
                            client: OpenAI) -> List[Dict[str, Any]]:
    """Create embeddings for a batch of texts.
    
    Args:
        texts: List of text chunks with metadata
        client: OpenAI client instance
        
    Returns:
        List of data entries with embeddings ready for Milvus
    """
    milvus_data = []
    for chunk in tqdm(texts, desc="Creating embeddings"):
        embedding = create_embedding(chunk["text"], client)
        milvus_data.append({
            "text": chunk["text"],
            "source": chunk["metadata"]["source"],
            "vector": embedding
        })
    return milvus_data


def find_document_files() -> List[str]:
    """Find all document files that need to be processed.
    
    Returns:
        List of file paths to process
    """
    logger.info("Finding all files to process...")
    file_pattern = ["/workspace/files/**/*.txt", "/workspace/files/**/*.md"]
    files_to_process = []
    for pattern in file_pattern:
        files_to_process.extend(glob.glob(pattern, recursive=True))

    logger.info(f"Found {len(files_to_process)} files to process")
    return files_to_process


def stream_read_file(
        file_path: str,
        block_size: int = FILE_BLOCK_SIZE) -> Generator[str, None, None]:
    """Stream read a file in chunks to avoid loading large files entirely into memory.
    
    Args:
        file_path: Path to the file to read
        block_size: Size of each block to read
        
    Yields:
        Chunks of the file content
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            while True:
                block = f.read(block_size)
                if not block:
                    break
                yield block
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")


def stream_process_file(
        file_path: str) -> Generator[Dict[str, Any], None, None]:
    """Stream process a single file and yield text chunks.
    
    Args:
        file_path: Path to the file to process
        
    Yields:
        Chunk objects with text and metadata
    """
    relative_path = os.path.relpath(file_path, "/workspace/files/")
    logger.info(f"Processing file: {relative_path}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    buffer = ""
    chunk_count = 0

    # Read the file in blocks and process incrementally
    for block in stream_read_file(file_path):
        buffer += block

        if len(buffer) >= CHUNK_SIZE + CHUNK_OVERLAP:
            # Create documents from the buffer
            chunks = text_splitter.split_text(buffer)

            # Yield all documents except the last one
            for chunk in chunks[:-1]:
                chunk_count += 1
                yield {"text": chunk, "metadata": {"source": relative_path}}

            # Keep the last document as the buffer for next iteration
            if chunks:
                buffer = chunks[-1]
            else:
                buffer = ""

    # Process any remaining content in the buffer
    if buffer.strip():
        chunks = text_splitter.split_text(buffer)
        for chunk in chunks:
            chunk_count += 1
            yield {"text": chunk, "metadata": {"source": relative_path}}

    logger.info(f"Created {chunk_count} chunks from {relative_path}")


def process_chunks_in_batches(milvus_client: MilvusClient,
                              embedding_client: OpenAI) -> None:
    """Process chunks in batches, create embeddings, and store in Milvus.
    
    Args:
        milvus_client: Milvus client instance
        embedding_client: OpenAI client for embeddings
    """
    files = find_document_files()
    total_chunks_processed = 0
    batch_chunks = []

    for file_path in files:
        # Stream process each file
        for chunk in stream_process_file(file_path):
            batch_chunks.append(chunk)

            # When we have enough chunks, process and insert them
            if len(batch_chunks) >= CHUNK_BATCH_SIZE:
                logger.info(f"Processing batch of {len(batch_chunks)} chunks")

                # Create embeddings for this batch
                milvus_data = create_embeddings(batch_chunks,
                                                      embedding_client)

                # Insert into Milvus
                if milvus_data:
                    logger.info(
                        f"Inserting {len(milvus_data)} chunks into Milvus collection {COLLECTION_NAME}"
                    )
                    milvus_client.insert(COLLECTION_NAME, milvus_data)

                # Update counts and clear batch
                total_chunks_processed += len(batch_chunks)
                logger.info(
                    f"Total chunks processed so far: {total_chunks_processed}")
                batch_chunks = []

    # Process any remaining chunks in the final batch
    if batch_chunks:
        logger.info(f"Processing final batch of {len(batch_chunks)} chunks")
        milvus_data = create_embeddings(batch_chunks, embedding_client)

        if milvus_data:
            logger.info(
                f"Inserting {len(milvus_data)} chunks into Milvus collection {COLLECTION_NAME}"
            )
            milvus_client.insert(COLLECTION_NAME, milvus_data)

        total_chunks_processed += len(batch_chunks)

    logger.info(f"Total chunks processed: {total_chunks_processed}")


def main():
    """Main function to run the data processing and insertion workflow."""
    logger.info("=== Starting insert-data workflow ===")

    try:
        # Initialize clients
        logger.info("Initializing clients...")
        embedding_client = OpenAI(base_url=EMBEDDING_BASE_URL, api_key="dummy")
        milvus_client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
        logger.info("Clients initialized successfully")

        # Setup Milvus collection
        logger.info("Setting up Milvus collection...")
        setup_milvus(milvus_client)

        # Process chunks in batches
        logger.info("Starting chunk processing in batches...")
        process_chunks_in_batches(milvus_client, embedding_client)

        logger.info("=== insert-data workflow completed successfully ===")
    except Exception as e:
        logger.error(f"Workflow failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
