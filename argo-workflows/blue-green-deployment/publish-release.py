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
"""

import os
import glob
from typing import List, Dict, Any, Tuple
import logging
import torch
from PIL import Image
import timm
from tqdm import tqdm
from openai import OpenAI
from pymilvus import MilvusClient, DataType

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
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

# Constants
CHUNK_SIZE = 512  # Token size for each chunk
CHUNK_OVERLAP = 64  # Token overlap between chunks
IMAGE_SIZE = 384  # Input size for the image model
BATCH_SIZE = 16  # Batch size for image processing


def load_text_documents() -> Dict[str, str]:
    """Load text document files from workspace/files directory.
    
    Returns:
        Dict mapping filenames to file contents
    """
    file_contents = {}

    # Read the text file list if it exists
    file_list_path = "/workspace/s3_text_file_list.txt"
    if os.path.exists(file_list_path):
        logger.info(f"Reading text file list from {file_list_path}")

    # Process all text files (full deployment)
    logger.info("Processing all text files...")
    file_pattern = ["/workspace/files/**/*.txt", "/workspace/files/**/*.md"]
    files_to_process = []
    for pattern in file_pattern:
        files_to_process.extend(glob.glob(pattern, recursive=True))

    logger.info(f"Found {len(files_to_process)} text files to process")

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

    return files_to_process


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
    try:
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
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {e}")
        # Return a zero tensor as fallback
        return torch.zeros(1, 3, image_size, image_size)


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
    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i:i + BATCH_SIZE]
        batch_tensors = []

        # Preprocess each image
        for path in batch_paths:
            img_tensor = preprocess_image(path)
            batch_tensors.append(img_tensor)

        # Combine into a batch
        if batch_tensors:
            batch = torch.cat(batch_tensors, dim=0).to(device)

            # Generate embeddings
            with torch.no_grad():
                batch_embeddings = model(batch)

            # Convert to Python lists and normalize
            for emb in batch_embeddings:
                # L2 normalize the embedding vector
                norm_emb = emb / torch.norm(emb)
                embeddings.append(norm_emb.cpu().tolist())

    return embeddings


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
        response = client.embeddings.create(input=text,
                                            model=TEXT_EMBEDDING_MODEL)
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error creating embedding: {e}")
        # Return a zero vector as fallback
        return [0.0] * TEXT_EMBEDDING_DIM


def create_database(client: MilvusClient, database_name: str) -> None:
    """Create a new Milvus database.
    
    Args:
        client: Milvus client instance
        database_name: Name of the database to create
    """
    logger.info(f"Creating database: {database_name}")

    try:
        # List existing databases to check if it already exists
        databases = client.list_databases()

        if database_name in databases:
            logger.error(f"Database {database_name} already exists")
            raise RuntimeError(
                f"Database {database_name} already exists. Please drop the database or change the database name manually."
            )
        else:
            # Create the database
            client.create_database(database_name)
            logger.info(f"Database {database_name} created successfully")
            client.using_database(database_name)
    except Exception as e:
        logger.error(f"Error creating database {database_name}: {e}")
        raise


def setup_collection(client: MilvusClient,
                     database_name: str,
                     collection_name: str,
                     is_image_collection: bool = False) -> str:
    """Setup Milvus collection in the specified database.
    
    Args:
        client: Milvus client instance
        database_name: Name of the database
        collection_name: Name of the collection to create
        is_image_collection: Whether this is an image collection (determines embedding dimension)
        
    Returns:
        Fully qualified collection name (database.collection)
    """
    # Create a fully qualified collection name
    full_collection_name = f"{database_name}.{collection_name}"

    logger.info(f"Setting up collection: {full_collection_name}")

    # Select embedding dimension based on collection type
    embedding_dim = IMAGE_EMBEDDING_DIM if is_image_collection else TEXT_EMBEDDING_DIM

    # Create schema
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
                     dim=embedding_dim)

    # Set up index parameters
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="IP"  # Inner product distance
    )

    logger.info(
        f"Creating collection {full_collection_name} with vector dimension {embedding_dim}"
    )
    client.create_collection(collection_name=collection_name,
                             schema=schema,
                             index_params=index_params)

    return full_collection_name


def process_text_documents(documents: Dict[str,
                                           str], milvus_client: MilvusClient,
                           embedding_client: OpenAI, database_name: str,
                           collection_name: str) -> None:
    """Process text documents, create chunks, and store in Milvus.
    
    Args:
        documents: Dictionary of documents {filename: content}
        milvus_client: Milvus client instance
        embedding_client: OpenAI client for embeddings
        database_name: Name of the database
        collection_name: Name of the collection within the database
    """
    if not documents:
        logger.warning("No text documents to process")
        return

    all_chunks = []
    full_collection_name = f"{database_name}.{collection_name}"

    logger.info(f"Processing {len(documents)} text documents")
    for filename, content in documents.items():
        logger.info(f"Processing text document: {filename}")
        chunks = chunk_text(content, filename)
        all_chunks.extend(chunks)

    logger.info(f"Created {len(all_chunks)} chunks")

    # Create embeddings and prepare data for Milvus
    milvus_data = []
    for i, chunk in enumerate(
            tqdm(all_chunks, desc="Creating embeddings for text")):
        embedding = create_embedding(chunk["text"], embedding_client)
        milvus_data.append({
            "text": chunk["text"],
            "source": chunk["metadata"]["source"],
            "vector": embedding
        })

    # Insert data into Milvus
    if milvus_data:
        logger.info(
            f"Inserting {len(milvus_data)} text documents into collection {full_collection_name}"
        )
        milvus_client.insert(collection_name=collection_name, data=milvus_data)
        logger.info("Text data insertion complete")
    else:
        logger.warning("No text data to insert")


def process_image_documents(image_files: List[str],
                            milvus_client: MilvusClient,
                            embedding_client: OpenAI, database_name: str,
                            collection_name: str) -> None:
    """Process image files and store in Milvus.
    
    Args:
        image_files: List of image file paths
        milvus_client: Milvus client instance
        embedding_client: OpenAI client for embeddings (not used for images)
        database_name: Name of the database
        collection_name: Name of the collection within the database
    """
    if not image_files:
        logger.warning("No image files to process")
        return

    logger.info(f"Processing {len(image_files)} image files")
    full_collection_name = f"{database_name}.{collection_name}"

    # Load the image embedding model
    image_model = load_image_model()

    # Generate embeddings for all images
    logger.info(
        f"Generating embeddings for images using {IMAGE_EMBEDDING_MODEL} model..."
    )
    all_embeddings = create_image_embeddings(image_files, image_model)

    # Prepare data for Milvus
    milvus_data = []
    for i, image_path in enumerate(
            tqdm(image_files, desc="Preparing image data")):
        relative_path = os.path.relpath(image_path, "/workspace/files/")

        if i < len(all_embeddings):
            milvus_data.append({
                "text": f"Image file: {relative_path}",
                "source": relative_path,
                "vector": all_embeddings[i]
            })

    # Insert data into Milvus
    if milvus_data:
        logger.info(
            f"Inserting {len(milvus_data)} image documents into collection {full_collection_name}"
        )
        milvus_client.insert(collection_name=collection_name, data=milvus_data)
        logger.info("Image data insertion complete")
    else:
        logger.warning("No image data to insert")


def main():
    """Main function to run the data processing and insertion workflow."""
    if not DATABASE_NAME:
        logger.error("DATABASE_NAME is required but not provided")
        raise ValueError("DATABASE_NAME environment variable is required")

    logger.info(
        f"Starting blue-green deployment workflow with database: {DATABASE_NAME}"
    )

    # Initialize clients
    embedding_client = OpenAI(base_url=TEXT_EMBEDDING_BASE_URL,
                              api_key="dummy")
    milvus_client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)

    # Create database for blue-green deployment
    create_database(milvus_client, DATABASE_NAME)

    # Process text documents
    if COLLECTION_NAME_TEXT:
        logger.info(
            f"Processing text documents for collection {DATABASE_NAME}.{COLLECTION_NAME_TEXT}"
        )
        # Setup collection for text in the database
        setup_collection(milvus_client,
                         DATABASE_NAME,
                         COLLECTION_NAME_TEXT,
                         is_image_collection=False)

        # Load and process text documents
        text_documents = load_text_documents()
        process_text_documents(text_documents, milvus_client, embedding_client,
                               DATABASE_NAME, COLLECTION_NAME_TEXT)
    else:
        logger.info(
            "Skipping text document processing (no collection name provided)")

    # Process image documents
    if COLLECTION_NAME_IMAGE:
        logger.info(
            f"Processing image documents for collection {DATABASE_NAME}.{COLLECTION_NAME_IMAGE}"
        )
        # Setup collection for images in the database with appropriate dimension
        setup_collection(milvus_client,
                         DATABASE_NAME,
                         COLLECTION_NAME_IMAGE,
                         is_image_collection=True)

        # Load and process image documents
        image_files = load_image_documents()
        process_image_documents(image_files, milvus_client, embedding_client,
                                DATABASE_NAME, COLLECTION_NAME_IMAGE)
    else:
        logger.info(
            "Skipping image document processing (no collection name provided)")

    logger.info(
        f"Blue-green deployment completed successfully for database: {DATABASE_NAME}"
    )


if __name__ == "__main__":
    main()
