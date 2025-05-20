#!/usr/bin/env python3
"""
Script to process text and image files, chunk them, create embeddings, and store them in Milvus.
This script implements the blue-green deployment workflow for inserting data into Milvus collections.

Environment variables:
- DATABASE_NAME: Name of the Milvus database to create
- COLLECTION_NAME_TEXT: Name of the collection for text data (will be created in DATABASE_NAME)
- COLLECTION_NAME_IMAGE: Name of the collection for image data (will be created in DATABASE_NAME)
- MILVUS_URI: Milvus connection URI
- TEXT_EMBEDDING_BASE_URL: Base URL for the text embedding model API
- TEXT_EMBEDDING_MODEL: Name of the text embedding model to use
- TEXT_EMBEDDING_DIM: Dimension of the text embedding vectors
- IMAGE_EMBEDDING_MODEL: Name of the image embedding model to use
- IMAGE_EMBEDDING_DIM: Dimension of the image embedding vectors
- LOG_FILE: Optional path to the log file
"""

import os
import glob
import sys
from typing import List, Dict, Any, Tuple, Generator
import logging
from dotenv import load_dotenv
import torch
from PIL import Image
import timm
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
MILVUS_URI = os.environ.get("MILVUS_URI")
MILVUS_TOKEN = os.environ.get("MILVUS_TOKEN")
DATABASE_NAME = os.environ.get("DATABASE_NAME")
COLLECTION_NAME_TEXT = os.environ.get("COLLECTION_NAME_TEXT",
                                      "text_collection")
COLLECTION_NAME_IMAGE = os.environ.get("COLLECTION_NAME_IMAGE",
                                       "image_collection")
TEXT_EMBEDDING_BASE_URL = os.environ.get("TEXT_EMBEDDING_BASE_URL")
TEXT_EMBEDDING_MODEL = os.environ.get("TEXT_EMBEDDING_MODEL")
TEXT_EMBEDDING_DIM = int(os.environ.get("TEXT_EMBEDDING_DIM"))
IMAGE_EMBEDDING_MODEL = os.environ.get(
    "IMAGE_EMBEDDING_MODEL", "vit_so400m_patch16_siglip_384.v2_webli")
IMAGE_EMBEDDING_DIM = int(
    os.environ.get("IMAGE_EMBEDDING_DIM",
                   "1152"))  # ViT-SigLIP model outputs 1152-dim vectors

# Log environment variables
logger.info("=== publish-release.py environment ===")
logger.info(f"DATABASE_NAME: {DATABASE_NAME}")
logger.info(f"COLLECTION_NAME_TEXT: {COLLECTION_NAME_TEXT}")
logger.info(f"COLLECTION_NAME_IMAGE: {COLLECTION_NAME_IMAGE}")
logger.info(f"MILVUS_URI: {MILVUS_URI}")
logger.info(f"TEXT_EMBEDDING_BASE_URL: {TEXT_EMBEDDING_BASE_URL}")
logger.info(f"TEXT_EMBEDDING_MODEL: {TEXT_EMBEDDING_MODEL}")
logger.info(f"TEXT_EMBEDDING_DIM: {TEXT_EMBEDDING_DIM}")
logger.info(f"IMAGE_EMBEDDING_MODEL: {IMAGE_EMBEDDING_MODEL}")
logger.info(f"IMAGE_EMBEDDING_DIM: {IMAGE_EMBEDDING_DIM}")
logger.info(f"LOG_FILE: {log_file}")

# Constants
CHUNK_SIZE = 512  # Token size for each chunk
CHUNK_OVERLAP = 64  # Token overlap between chunks
CHUNK_BATCH_SIZE = 100  # Batch size for text processing
FILE_BLOCK_SIZE = 1048576  # Block size for file reading
IMAGE_SIZE = 384  # Input size for the image model
IMAGE_BATCH_SIZE = 16  # Batch size for image processing


def setup_collection(client: MilvusClient,
                     database_name: str,
                     collection_name: str,
                     is_image_collection: bool = False) -> str:
    """Setup Milvus collection using blue-green deployment pattern.
    
    Args:
        client: Milvus client instance
        database_name: Name of the database
        collection_name: Base name of the collection
        is_image_collection: Whether the collection is for image data
        
    Returns:
        The actual collection name created (suffixed with timestamp)
    """
    # Select appropriate database
    client.using_database(database_name)
    logger.info(f"Using database: {database_name}")
    logger.info(f"Creating collection: {collection_name}")

    # Determine embedding dimension based on collection type
    embedding_dim = IMAGE_EMBEDDING_DIM if is_image_collection else TEXT_EMBEDDING_DIM

    # Create schema
    schema = MilvusClient.create_schema()
    schema.add_field(field_name="id",
                     datatype=DataType.INT64,
                     is_primary=True,
                     auto_id=True)
    if not is_image_collection:
        schema.add_field(field_name="text",
                         datatype=DataType.VARCHAR,
                         max_length=65535)
    schema.add_field(field_name="source",
                     datatype=DataType.VARCHAR,
                     max_length=255)
    schema.add_field(field_name="vector",
                     datatype=DataType.FLOAT_VECTOR,
                     dim=embedding_dim)

    # Set up index parameters
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="IP"  # Inner product distance
    )

    # Create collection
    client.create_collection(collection_name=collection_name,
                             schema=schema,
                             index_params=index_params)
    logger.info(f"Collection {collection_name} created successfully")

    return collection_name


def create_database(client: MilvusClient, database_name: str) -> None:
    """Create a Milvus database.
    
    Args:
        client: Milvus client instance
        database_name: Name of the database to create
    """
    # Check if database exists
    databases = client.list_databases()
    if database_name in databases:
        logger.info(f"Database {database_name} already exists")
        return

    # Create database
    logger.info(f"Creating database {database_name}")
    client.create_database(database_name)
    logger.info(f"Database {database_name} created successfully")


def load_image_model() -> torch.nn.Module:
    """Load the pre-trained image embedding model.
    
    Returns:
        A pre-trained image embedding model
    """
    logger.info(
        f"Loading {IMAGE_EMBEDDING_MODEL} model for image embeddings...")
    try:
        # Set device for model loading and inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Create the model with pretrained weights, removing the classification head
        model = timm.create_model(IMAGE_EMBEDDING_MODEL,
                                  pretrained=True,
                                  num_classes=0).to(device)

        # Set to evaluation mode
        model.eval()

        logger.info("Image model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading image model: {e}")
        raise


def preprocess_image(image_path: str,
                     image_size: int = IMAGE_SIZE) -> torch.Tensor:
    """Preprocess an image for the embedding model.
    
    Args:
        image_path: Path to the image file
        image_size: Size to resize the image to
        
    Returns:
        Preprocessed image tensor
    """
    # Open the image file
    img = Image.open(image_path).convert('RGB')

    # Resize and center crop
    img = img.resize((image_size, image_size), Image.LANCZOS)

    # Convert to tensor and normalize
    img_tensor = torch.tensor(list(img.getdata()),
                              dtype=torch.float32).reshape(
                                  1, 3, image_size, image_size)

    # Normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    img_tensor = (img_tensor / 255.0 - mean) / std

    return img_tensor


def create_image_embeddings(image_paths: List[str],
                            model: torch.nn.Module) -> List[List[float]]:
    """Create embeddings for a batch of images using the local model.
    
    Args:
        image_paths: Paths to the image files
        model: The pre-trained image model
        
    Returns:
        List of image embedding vectors
    """
    if not image_paths:
        return []

    device = next(model.parameters()).device
    embeddings = []

    # Process images in batches
    for i in range(0, len(image_paths), IMAGE_BATCH_SIZE):
        batch_paths = image_paths[i:i + IMAGE_BATCH_SIZE]
        batch_tensors = []

        for image_path in batch_paths:
            img_tensor = preprocess_image(image_path).to(device)
            batch_tensors.append(img_tensor)

        if batch_tensors:
            # Stack tensors into a batch
            batch_input = torch.cat(batch_tensors, dim=0)

            # Compute embeddings
            with torch.no_grad():
                batch_output = model(batch_input)
                for j in range(len(batch_paths)):
                    embedding = batch_output[j].cpu().numpy().tolist()
                    embeddings.append(embedding)

    return embeddings


def create_embedding(text: str, client: OpenAI) -> List[float]:
    """Create embedding for text using OpenAI API.
    
    Args:
        text: Input text to embed
        client: OpenAI client instance
        
    Returns:
        Embedding vector
    """
    response = client.embeddings.create(input=text, model=TEXT_EMBEDDING_MODEL)
    return response.data[0].embedding


def create_text_embeddings(texts: List[Dict[str, Any]],
                        client: OpenAI) -> List[Dict[str, Any]]:
    """Create embeddings for a batch of texts.
    
    Args:
        texts: List of text chunks with metadata
        client: OpenAI client instance
        
    Returns:
        List of data entries with embeddings ready for Milvus
    """
    milvus_data = []
    for chunk in tqdm(texts, desc="Creating text embeddings"):
        embedding = create_embedding(chunk["text"], client)
        milvus_data.append({
            "text": chunk["text"],
            "source": chunk["metadata"]["source"],
            "vector": embedding
        })
    return milvus_data


def load_text_documents() -> List[str]:
    """Find all text document files from workspace/files directory.
    
    Returns:
        List of file paths to process
    """
    # Process all text files
    logger.info("Finding all text files to process...")
    file_pattern = ["/workspace/files/**/*.txt", "/workspace/files/**/*.md"]
    files_to_process = []
    for pattern in file_pattern:
        files_to_process.extend(glob.glob(pattern, recursive=True))

    logger.info(f"Found {len(files_to_process)} text files to process")
    return files_to_process


def load_image_documents() -> List[str]:
    """Load image files from workspace/files directory.
    
    Returns:
        List of image file paths
    """
    # Process all image files
    logger.info("Processing all image files...")
    file_pattern = [
        "/workspace/files/**/*.jpg", "/workspace/files/**/*.jpeg",
        "/workspace/files/**/*.png"
    ]
    files_to_process = []
    for pattern in file_pattern:
        files_to_process.extend(glob.glob(pattern, recursive=True))

    logger.info(f"Found {len(files_to_process)} image files to process")
    for image_path in files_to_process[:
                                       5]:  # Log first 5 image files for verification
        logger.debug(
            f"Image file: {os.path.relpath(image_path, '/workspace/files/')}")

    return files_to_process


def stream_read_file(file_path: str,
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


def process_text_documents(text_files: List[str], milvus_client: MilvusClient,
                           embedding_client: OpenAI, database_name: str,
                           collection_name: str) -> None:
    """Process text documents, create chunks, and store in Milvus using batch processing.
    
    Args:
        text_files: List of text file paths
        milvus_client: Milvus client instance
        embedding_client: OpenAI client for embeddings
        database_name: Database to use
        collection_name: Collection to insert data into
    """
    # Use specified database
    milvus_client.using_database(database_name)

    if not text_files:
        logger.warning("No text files to process")
        return

    logger.info(f"Processing {len(text_files)} text documents in batches")
    
    batch_chunks = []
    total_chunks_processed = 0
    
    # Process each file
    for file_path in text_files:
        # Stream process each file
        for chunk in stream_process_file(file_path):
            batch_chunks.append(chunk)

            # When we have enough chunks, process and insert them
            if len(batch_chunks) >= CHUNK_BATCH_SIZE:
                logger.info(f"Processing batch of {len(batch_chunks)} chunks")

                # Create embeddings for this batch
                milvus_data = create_text_embeddings(batch_chunks, embedding_client)

                # Insert into Milvus
                if milvus_data:
                    logger.info(
                        f"Inserting {len(milvus_data)} chunks into Milvus collection {collection_name}"
                    )
                    milvus_client.insert(collection_name, milvus_data)

                # Update counts and clear batch
                total_chunks_processed += len(batch_chunks)
                logger.info(
                    f"Total chunks processed so far: {total_chunks_processed}")
                batch_chunks = []

    # Process any remaining chunks in the final batch
    if batch_chunks:
        logger.info(f"Processing final batch of {len(batch_chunks)} chunks")
        milvus_data = create_text_embeddings(batch_chunks, embedding_client)

        if milvus_data:
            logger.info(
                f"Inserting {len(milvus_data)} chunks into Milvus collection {collection_name}"
            )
            milvus_client.insert(collection_name, milvus_data)

        total_chunks_processed += len(batch_chunks)

    logger.info(f"Total chunks processed: {total_chunks_processed}")


def process_image_documents(image_files: List[str],
                            milvus_client: MilvusClient,
                            embedding_client: OpenAI, database_name: str,
                            collection_name: str) -> None:
    """Process image documents, create embeddings, and store in Milvus.
    
    Args:
        image_files: List of image file paths
        milvus_client: Milvus client instance
        embedding_client: OpenAI client for embeddings
        database_name: Database to use
        collection_name: Collection to insert data into
    """
    # Use specified database
    milvus_client.using_database(database_name)

    if not image_files:
        logger.warning("No image files to process")
        return

    logger.info(f"Processing {len(image_files)} image files")

    # Load the image model
    model = load_image_model()

    # Create embeddings for all images
    logger.info("Creating embeddings for images")
    embeddings = create_image_embeddings(image_files, model)
    logger.info(f"Created {len(embeddings)} image embeddings")

    # Prepare data for Milvus
    milvus_data = []
    for i, image_path in enumerate(image_files):
        if i < len(embeddings):
            relative_path = os.path.relpath(image_path, "/workspace/files/")
            milvus_data.append({
                "source": relative_path,
                "vector": embeddings[i]
            })

    # Insert data into Milvus
    if milvus_data:
        logger.info(
            f"Inserting {len(milvus_data)} image records into Milvus collection {collection_name}"
        )
        milvus_client.insert(collection_name, milvus_data)
        logger.info("Image data insertion complete")
    else:
        logger.warning("No image data to insert")


def generate_release_summary(database_name: str,
                             text_collection: str = None,
                             image_collection: str = None,
                             text_count: int = 0,
                             image_count: int = 0) -> None:
    """Generate a summary of the release and write it to /workspace/release_summary.md.
    
    Args:
        database_name: Name of the database used
        text_collection: Name of the text collection created (or None if not created)
        image_collection: Name of the image collection created (or None if not created)
        text_count: Number of text documents processed
        image_count: Number of image files processed
    """
    logger.info("Generating release summary...")

    # Get current timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create summary content
    summary = f"""# Blue-Green Deployment Release Summary

## Overview
- **Release Date:** {timestamp}
- **Database:** {database_name}

## Collections
"""

    if text_collection:
        summary += f"- **Text Collection:** {text_collection}\n"
        summary += f"  - Documents Processed: {text_count}\n"
    else:
        summary += "- **Text Collection:** Not created\n"

    if image_collection:
        summary += f"- **Image Collection:** {image_collection}\n"
        summary += f"  - Images Processed: {image_count}\n"
    else:
        summary += "- **Image Collection:** Not created\n"

    summary += "\n## Processing Details\n"

    # Add text processing details
    if text_collection and text_count > 0:
        summary += "### Text Processing\n"
        summary += f"- Embedding Model: {TEXT_EMBEDDING_MODEL}\n"
        summary += f"- Embedding Dimension: {TEXT_EMBEDDING_DIM}\n"
        summary += f"- Chunk Size: {CHUNK_SIZE}\n"
        summary += f"- Chunk Overlap: {CHUNK_OVERLAP}\n\n"

    # Add image processing details
    if image_collection and image_count > 0:
        summary += "### Image Processing\n"
        summary += f"- Embedding Model: {IMAGE_EMBEDDING_MODEL}\n"
        summary += f"- Embedding Dimension: {IMAGE_EMBEDDING_DIM}\n"
        summary += f"- Image Size: {IMAGE_SIZE}\n\n"

    # Add S3 source information if available
    if os.path.exists("/env-config/env.properties"):
        with open("/env-config/env.properties", "r") as f:
            env_content = f.read()
            s3_path_text = None
            s3_path_image = None

            # Extract S3 paths from env file
            for line in env_content.splitlines():
                if line.startswith("S3_PATH_TEXT="):
                    s3_path_text = line.split(
                        "=", 1)[1].strip().strip('"').strip("'")
                elif line.startswith("S3_PATH_IMAGE="):
                    s3_path_image = line.split(
                        "=", 1)[1].strip().strip('"').strip("'")

            summary += "## Data Sources\n"
            if s3_path_text:
                summary += f"- Text Source: `{s3_path_text}`\n"
            if s3_path_image:
                summary += f"- Image Source: `{s3_path_image}`\n"

    # Add file lists statistics
    summary += "\n## File Statistics\n"

    if os.path.exists("/workspace/s3_text_file_list.txt"):
        with open("/workspace/s3_text_file_list.txt", "r") as f:
            text_file_count = sum(1 for _ in f)
            summary += f"- Text Files from S3: {text_file_count}\n"

    if os.path.exists("/workspace/s3_image_file_list.txt"):
        with open("/workspace/s3_image_file_list.txt", "r") as f:
            image_file_count = sum(1 for _ in f)
            summary += f"- Image Files from S3: {image_file_count}\n"

    # Write summary to file
    log_dir = os.environ.get("LOG_DIR")
    summary_path = os.path.join(
        log_dir,
        "release_summary.md") if log_dir else "/workspace/release_summary.md"
    with open(summary_path, "w") as f:
        f.write(summary)

    logger.info(f"Release summary generated and saved to {summary_path}")


def main():
    """Main function to run the blue-green deployment workflow."""
    logger.info("=== Starting publish-release workflow ===")

    text_count = 0
    image_count = 0

    try:
        # Initialize clients
        logger.info("Initializing clients...")
        text_embedding_client = OpenAI(base_url=TEXT_EMBEDDING_BASE_URL,
                                       api_key="dummy")
        milvus_client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
        logger.info("Clients initialized successfully")

        # Create database
        logger.info(f"Setting up database {DATABASE_NAME}...")
        create_database(milvus_client, DATABASE_NAME)

        # Process text documents if the text collection name is set
        if COLLECTION_NAME_TEXT:
            logger.info("Starting text document processing...")
            # Setup text collection
            text_collection = setup_collection(milvus_client,
                                                      DATABASE_NAME,
                                                      COLLECTION_NAME_TEXT)

            # Find and process text files
            text_files = load_text_documents()
            text_count = len(text_files)
            if text_files:
                process_text_documents(text_files, milvus_client,
                                       text_embedding_client, DATABASE_NAME,
                                       text_collection)
                logger.info(
                    f"Text collection published as: {text_collection}")
            else:
                logger.warning("No text documents found to process")
        else:
            logger.info(
                "COLLECTION_NAME_TEXT not set, skipping text processing")

        # Process image documents if the image collection name is set
        if COLLECTION_NAME_IMAGE:
            logger.info("Starting image document processing...")
            # Setup image collection
            image_collection = setup_collection(
                milvus_client,
                DATABASE_NAME,
                COLLECTION_NAME_IMAGE,
                is_image_collection=True)

            # Load and process image documents
            image_files = load_image_documents()
            image_count = len(image_files)
            if image_files:
                process_image_documents(image_files, milvus_client,
                                        text_embedding_client, DATABASE_NAME,
                                        image_collection)
                logger.info(
                    f"Image collection published as: {image_collection}"
                )
            else:
                logger.warning("No image files found to process")
        else:
            logger.info(
                "COLLECTION_NAME_IMAGE not set, skipping image processing")

        # Generate release summary
        generate_release_summary(database_name=DATABASE_NAME,
                                 text_collection=text_collection,
                                 image_collection=image_collection,
                                 text_count=text_count,
                                 image_count=image_count)

        logger.info("=== publish-release workflow completed successfully ===")
    except Exception as e:
        logger.error(f"Workflow failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
