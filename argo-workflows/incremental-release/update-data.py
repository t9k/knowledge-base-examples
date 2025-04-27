#!/usr/bin/env python3
"""
Script to process text files, chunk them, create embeddings, and store them in Milvus.
This script implements the full deployment workflow for inserting data into a Milvus collection.

Environment variables:
- DATABASE_NAME: Name of the Milvus database to use
- COLLECTION_NAME: Name of the Milvus collection to create/use
- MILVUS_URI: Milvus connection URI
- EMBEDDING_BASE_URL: Base URL for the embedding model API
- EMBEDDING_MODEL: Name of the embedding model to use
- EMBEDDING_DIM: Dimension of the embedding vectors
- LOG_FILE: Optional path to the log file
"""

import os
import glob
import datetime
import sys
from typing import List, Dict, Any
import logging
from dotenv import load_dotenv

from tqdm import tqdm
from openai import OpenAI
from pymilvus import MilvusClient, DataType

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
logger.info("=== update-data.py environment ===")
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
                    logger.debug(
                        f"Loaded file: {relative_path} ({len(content)} bytes)")
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")

    return file_contents


def chunk_text(text: str, filename: str,
               last_modified: str) -> List[Dict[str, Any]]:
    """Split text into chunks with metadata.
    
    Args:
        text: Text content to chunk
        filename: Source filename for metadata
        last_modified: Last modification time of the file
        
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
                        "source": filename,
                        "last_modified": last_modified
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
                "source": filename,
                "last_modified": last_modified
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
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response.data[0].embedding


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

    # Check if collection exists, create it if not
    if not client.has_collection(COLLECTION_NAME):
        logger.info(
            f"Collection {COLLECTION_NAME} does not exist, creating it...")
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
        schema.add_field(field_name="last_modified",
                         datatype=DataType.VARCHAR,
                         max_length=64)
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
    else:
        logger.info(f"Collection {COLLECTION_NAME} already exists")


def delete_by_source(source: str, milvus_client: MilvusClient) -> None:
    """Delete all records with the specified source from Milvus.
    
    Args:
        source: Source field value to match for deletion
        milvus_client: Milvus client instance
    """
    # Prepare expression to filter by source
    expr = f'source == "{source}"'

    try:
        # Delete all records matching the source
        result = milvus_client.delete(collection_name=COLLECTION_NAME,
                                      filter=expr)
        milvus_client.flush(COLLECTION_NAME)
        logger.info(
            f"Deleted {result['delete_count']} records with source: {source}")
    except Exception as e:
        logger.error(f"Error deleting records with source {source}: {e}")


def process_document(filename: str, content: str, milvus_client: MilvusClient,
                     embedding_client: OpenAI) -> None:
    """Process a single document, create chunks, and store in Milvus.
    
    Args:
        filename: Source filename
        content: Document content
        milvus_client: Milvus client instance
        embedding_client: OpenAI client for embeddings
    """
    logger.info(f"Processing document: {filename}")

    # Get last modified time of the file
    file_path = os.path.join("/workspace/files/", filename)
    last_modified = datetime.datetime.fromtimestamp(
        os.path.getmtime(file_path)).isoformat()
    logger.info(f"File last modified: {last_modified}")

    # First check if there are any entities with matching filename
    expr = f'source == "{filename}"'
    search_results = milvus_client.query(collection_name=COLLECTION_NAME,
                                         filter=expr,
                                         output_fields=["id", "last_modified"],
                                         limit=1)

    # If no entities match the filename, proceed with normal processing
    if not search_results:
        logger.info(
            f"No existing entities found for {filename}, proceeding with full processing"
        )
    else:
        # Check if there are entities with matching filename AND last_modified
        expr = f'source == "{filename}" AND last_modified == "{last_modified}"'
        exact_match_results = milvus_client.query(
            collection_name=COLLECTION_NAME,
            filter=expr,
            output_fields=["id"],
            limit=1)

        # If entities with matching filename AND last_modified exist, skip processing
        if exact_match_results:
            logger.info(
                f"Found existing entities for {filename} with matching last_modified time, skipping processing"
            )
            return

        # If entities with matching filename exist but none with matching last_modified,
        # delete entities with matching filename
        logger.info(
            f"Found existing entities for {filename} but with different last_modified time, deleting and reprocessing"
        )
        delete_by_source(filename, milvus_client)

    # Create chunks from the document
    chunks = chunk_text(content, filename, last_modified)
    logger.info(f"Created {len(chunks)} chunks from {filename}")

    # Create embeddings and prepare data for Milvus
    milvus_data = []
    for chunk in tqdm(chunks, desc=f"Creating embeddings for {filename}"):
        embedding = create_embedding(chunk["text"], embedding_client)
        milvus_data.append({
            "text": chunk["text"],
            "source": chunk["metadata"]["source"],
            "last_modified": chunk["metadata"]["last_modified"],
            "vector": embedding
        })

    # Insert data into Milvus
    if milvus_data:
        logger.info(
            f"Inserting {len(milvus_data)} documents from {filename} into Milvus collection {COLLECTION_NAME}"
        )
        milvus_client.insert(COLLECTION_NAME, milvus_data)
        logger.info(f"Data insertion for {filename} complete")
    else:
        logger.warning(f"No data to insert for {filename}")


def main():
    """Main function to run the data processing and insertion workflow."""
    logger.info("=== Starting update-data.py workflow ===")

    try:
        # Initialize clients
        logger.info("Initializing clients...")
        embedding_client = OpenAI(base_url=EMBEDDING_BASE_URL, api_key="dummy")
        milvus_client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
        logger.info("Clients initialized successfully")

        # Setup Milvus collection
        logger.info("Setting up Milvus collection...")
        setup_milvus(milvus_client)

        # Load documents
        logger.info("Loading documents...")
        documents = load_documents()
        logger.info(f"Loaded {len(documents)} documents")

        # Process each document individually
        for filename, content in documents.items():
            logger.info(
                f"Processing document {filename} ({len(content)} bytes)")
            process_document(filename, content, milvus_client,
                             embedding_client)

        logger.info("=== update-data.py workflow completed successfully ===")
    except Exception as e:
        logger.error(f"Workflow failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
