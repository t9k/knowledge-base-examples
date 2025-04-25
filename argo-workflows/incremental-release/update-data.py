#!/usr/bin/env python3
"""
Script to incrementally process text files, chunk them, create embeddings, and update them in Milvus.
This script implements the incremental deployment workflow for updating data in a Milvus collection.

Environment variables:
- COLLECTION_NAME: Name of the Milvus collection to use
- MILVUS_URI: Milvus connection URI
- EMBEDDING_BASE_URL: Base URL for the embedding model API
- EMBEDDING_MODEL: Name of the embedding model to use
- EMBEDDING_DIM: Dimension of the embedding vectors
"""

import os
import shutil
from typing import List, Dict, Any, Tuple
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
LOG_FILE = "/workspace/log.txt"


def log_message(message: str) -> None:
    """Log message to both logger and log file.
    
    Args:
        message: Message to log
    """
    logger.info(message)
    with open(LOG_FILE, "a") as f:
        f.write(f"{message}\n")


def parse_change_log() -> Tuple[List[str], List[str], List[str]]:
    """Parse the log file to identify files to create, modify, and delete.
    Only parses log entries from the last sync run (after the last SYNC_BOUNDARY).
    
    Returns:
        Tuple of (files_to_create, files_to_modify, files_to_delete)
    """
    files_to_create = []
    files_to_modify = []
    files_to_delete = []

    try:
        with open(LOG_FILE, "r") as f:
            log_content = f.read()
            
        # Split by sync boundary marker and get the last section
        sections = log_content.split("====================SYNC_BOUNDARY====================")
        if not sections:
            logger.warning("No sync boundaries found in log file")
            return files_to_create, files_to_modify, files_to_delete
            
        # Get the last section (most recent sync run)
        last_section = sections[-1]
        
        # Process each line in the last section
        for line in last_section.splitlines():
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("TO_CREATE: "):
                files_to_create.append(line[11:])  # Remove 'TO_CREATE: ' prefix
            elif line.startswith("TO_MODIFY: "):
                files_to_modify.append(line[11:])  # Remove 'TO_MODIFY: ' prefix
            elif line.startswith("TO_DELETE: "):
                files_to_delete.append(line[11:])  # Remove 'TO_DELETE: ' prefix
    except Exception as e:
        logger.error(f"Error parsing change log: {e}")

    return files_to_create, files_to_modify, files_to_delete


def load_documents(file_list: List[str]) -> Dict[str, str]:
    """Load specific document files from workspace/files directory.
    
    Args:
        file_list: List of file paths to load
        
    Returns:
        Dict mapping filenames to file contents
    """
    file_contents = {}

    logger.info(f"Loading {len(file_list)} documents...")
    
    for file_path in file_list:
        try:
            full_path = os.path.join("/workspace/files/", file_path)
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
                if content:
                    file_contents[file_path] = content
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


def ensure_collection_exists(client: MilvusClient) -> None:
    """Ensure Milvus collection exists.
    
    Args:
        client: Milvus client instance
    """
    # Check if collection exists, create it if not
    if not client.has_collection(COLLECTION_NAME):
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
        log_message(f"Created new collection: {COLLECTION_NAME}")
    else:
        logger.info(f"Using existing collection {COLLECTION_NAME}")


def insert_document(filename: str, content: str, milvus_client: MilvusClient,
                   embedding_client: OpenAI) -> None:
    """Process a single document, create chunks, and insert in Milvus.
    
    Args:
        filename: File name to process
        content: Document content
        milvus_client: Milvus client instance
        embedding_client: OpenAI client for embeddings
    """
    if not content:
        log_message(f"Empty content for {filename}, skipping")
        return
        
    log_message(f"Processing document for insertion: {filename}")
    chunks = chunk_text(content, filename)
    
    if not chunks:
        log_message(f"No chunks created for {filename}, skipping")
        return
        
    log_message(f"Created {len(chunks)} chunks for {filename}")

    # Create embeddings and prepare data for Milvus
    milvus_data = []
    for chunk in tqdm(chunks, desc=f"Creating embeddings for {filename}"):
        embedding = create_embedding(chunk["text"], embedding_client)
        milvus_data.append({
            "text": chunk["text"],
            "source": chunk["metadata"]["source"],
            "vector": embedding
        })

    # Insert data into Milvus
    if milvus_data:
        log_message(f"Inserting {len(milvus_data)} chunks for {filename} into Milvus collection {COLLECTION_NAME}")
        milvus_client.insert(COLLECTION_NAME, milvus_data)
        
        # 强制刷新集合以确保插入生效
        try:
            milvus_client.flush(COLLECTION_NAME)
            log_message(f"Flushed collection after insertion")
        except Exception as e:
            logger.warning(f"Failed to flush collection after insertion: {e}")
            
        log_message(f"CREATED: {filename}")
    else:
        log_message(f"No data to insert for {filename}")


def delete_document(filename: str, milvus_client: MilvusClient) -> None:
    """Delete a document from Milvus based on source filename.
    
    Args:
        filename: Filename to delete
        milvus_client: Milvus client instance
    """
    log_message(f"Deleting data for file: {filename}")
    
    # 正确构造删除表达式，确保完全匹配
    expr = f'source == "{filename}"'
    
    try:
        # 检查是否有匹配的记录
        search_results = milvus_client.query(
            collection_name=COLLECTION_NAME,
            filter=expr,
            output_fields=["id"],
            limit=1
        )
        
        # 如果有匹配记录，执行删除
        if search_results:
            # 执行删除操作
            result = milvus_client.delete(
                collection_name=COLLECTION_NAME,
                filter=expr
            )
            
            # 强制刷新集合以确保删除生效
            try:
                milvus_client.flush(COLLECTION_NAME)
                log_message(f"Flushed collection after deletion")
            except Exception as e:
                logger.warning(f"Failed to flush collection after deletion: {e}")
            
            log_message(f"Deleted {result['delete_count']} rows for {filename}")
            log_message(f"DELETED: {filename}")
        else:
            log_message(f"No records found for {filename}, nothing to delete")
    except Exception as e:
        logger.error(f"Error deleting data for {filename}: {e}")
        log_message(f"Failed to delete {filename}: {str(e)}")


def update_document(filename: str, content: str, milvus_client: MilvusClient,
                   embedding_client: OpenAI) -> None:
    """Update a document by first deleting it and then inserting new version.
    
    Args:
        filename: Filename to update
        content: New document content
        milvus_client: Milvus client instance
        embedding_client: OpenAI client for embeddings
    """
    log_message(f"Updating data for file: {filename}")
    
    # 先删除旧数据
    delete_document(filename, milvus_client)
    
    # 再插入新数据（insert_document内部会添加CREATED日志）
    insert_document(filename, content, milvus_client, embedding_client)
    
    # 将CREATED日志修改为MODIFIED日志
    log_message(f"MODIFIED: {filename}")


def main():
    """Main function to run the incremental data update workflow."""
    log_message("Starting update-data workflow")

    # Initialize clients
    embedding_client = OpenAI(base_url=EMBEDDING_BASE_URL, api_key="dummy")
    milvus_client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)

    # Ensure collection exists
    ensure_collection_exists(milvus_client)

    # Parse change log to identify files to create, modify, and delete
    files_to_create, files_to_modify, files_to_delete = parse_change_log()
    
    log_message(f"Found {len(files_to_create)} files to create, {len(files_to_modify)} files to modify, and {len(files_to_delete)} files to delete")

    # Process files to create
    for filename in files_to_create:
        try:
            full_path = os.path.join("/workspace/files/", filename)
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
                insert_document(filename, content, milvus_client, embedding_client)
        except Exception as e:
            logger.error(f"Error processing file {filename} for creation: {e}")
    
    # Process files to modify
    for filename in files_to_modify:
        try:
            full_path = os.path.join("/workspace/files/", filename)
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
                update_document(filename, content, milvus_client, embedding_client)
        except Exception as e:
            logger.error(f"Error processing file {filename} for modification: {e}")
    
    # Process files to delete
    for filename in files_to_delete:
        try:
            delete_document(filename, milvus_client)
        except Exception as e:
            logger.error(f"Error processing file {filename} for deletion: {e}")

    # Update the s3_files.txt with the new version
    try:
        shutil.move("/workspace/s3_files.txt.new", "/workspace/s3_files.txt")
        log_message("Updated s3_files.txt with new version")
    except Exception as e:
        logger.error(f"Error updating s3_files.txt: {e}")

    log_message("Workflow completed successfully")


if __name__ == "__main__":
    main()
