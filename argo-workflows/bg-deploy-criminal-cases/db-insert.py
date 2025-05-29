#!/usr/bin/env python3
"""
Script to insert criminal cases data into Milvus.
"""
import sys
import os
import json
import uuid
import logging
from dotenv import load_dotenv
from tqdm import tqdm
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus import (
    MilvusClient,
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch
import torch_gcu

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
COLLECTION_NAME = os.environ.get("COLLECTION_NAME")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 256))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 32))

# Log environment variables
logger.info("=== insert_data_cases.py environment ===")
logger.info(f"MILVUS_URI: {MILVUS_URI}")
logger.info(f"DATABASE_NAME: {DATABASE_NAME}")
logger.info(f"COLLECTION_NAME: {COLLECTION_NAME}")
logger.info(f"CHUNK_SIZE: {CHUNK_SIZE}")
logger.info(f"CHUNK_OVERLAP: {CHUNK_OVERLAP}")
logger.info(f"LOG_FILE: {log_file}")

BATCH_SIZE = 100


def setup_milvus_collection(dense_dim):
    connections.connect(uri=MILVUS_URI, token=MILVUS_TOKEN, db_name=DATABASE_NAME)
    fields = [
        FieldSchema(name="chunk_id",
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    max_length=36),
        FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=3000),
        FieldSchema(name="relevant_articles",
                    dtype=DataType.ARRAY,
                    element_type=DataType.INT64,
                    max_capacity=9),
        FieldSchema(name="accusation",
                    dtype=DataType.ARRAY,
                    element_type=DataType.VARCHAR,
                    max_length=100,
                    max_capacity=13),
        FieldSchema(name="punish_of_money", dtype=DataType.INT64),
        FieldSchema(name="criminals",
                    dtype=DataType.ARRAY,
                    element_type=DataType.VARCHAR,
                    max_length=10,
                    max_capacity=30),
        FieldSchema(name="imprisonment", dtype=DataType.INT64),
        FieldSchema(name="life_imprisonment", dtype=DataType.BOOL),
        FieldSchema(name="death_penalty", dtype=DataType.BOOL),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="dense_vector",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=dense_dim),
    ]
    schema = CollectionSchema(fields)
    if utility.has_collection(COLLECTION_NAME):
        raise RuntimeError(f"Collection {COLLECTION_NAME} already exists")
    col = Collection(COLLECTION_NAME, schema, consistency_level="Strong")
    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
    col.create_index("sparse_vector", sparse_index)
    dense_index = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {
            "M": 24,
            "efConstruction": 400
        }
    }
    col.create_index("dense_vector", dense_index)
    col.load()
    return col


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


# 读取 jsonl 文件，每行一个 dict
def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\r\n", "\n", "。", "；", "，", "、"],
        keep_separator="end")
    return text_splitter.split_text(text)


def record_generator(jsonl_path):
    for item in read_jsonl(jsonl_path):
        fact = item["fact"].replace(",", "，").replace(";", "；")
        meta = item["meta"]

        chunks = chunk_text(fact) if len(fact) > CHUNK_SIZE else [fact]
        for chunk in chunks:
            if len(chunk) <= 3:
                continue

            metadata = {
                "relevant_articles":
                [int(i) for i in set(meta["relevant_articles"])],
                "accusation":
                meta["accusation"],
                "punish_of_money":
                meta["punish_of_money"],
                "criminals":
                meta["criminals"],
                "imprisonment":
                meta["term_of_imprisonment"]["imprisonment"],
                "life_imprisonment":
                meta["term_of_imprisonment"]["life_imprisonment"],
                "death_penalty":
                meta["term_of_imprisonment"]["death_penalty"]
            }

            yield {"chunk_id": str(uuid.uuid4()), "chunk": chunk, **metadata}


def insert_data_streaming(col: Collection,
                          record_iter,
                          ef,
                          batch_size=BATCH_SIZE):
    buffer = []
    total = 0
    for record in tqdm(record_iter, desc="Inserting records"):
        buffer.append(record)
        if len(buffer) >= batch_size:
            chunks = [r["chunk"] for r in buffer]
            embeddings = ef(chunks)
            to_insert = [
                [r["chunk_id"] for r in buffer],
                [r["chunk"] for r in buffer],
                [r["relevant_articles"] for r in buffer],
                [r["accusation"] for r in buffer],
                [r["punish_of_money"] for r in buffer],
                [r["criminals"] for r in buffer],
                [r["imprisonment"] for r in buffer],
                [r["life_imprisonment"] for r in buffer],
                [r["death_penalty"] for r in buffer],
                embeddings["sparse"],
                embeddings["dense"],
            ]
            col.insert(to_insert)
            total += len(buffer)
            buffer = []
    # 插入剩余部分
    if buffer:
        chunks = [r["chunk"] for r in buffer]
        embeddings = ef(chunks)
        to_insert = [
            [r["chunk_id"] for r in buffer],
            [r["chunk"] for r in buffer],
            [r["relevant_articles"] for r in buffer],
            [r["accusation"] for r in buffer],
            [r["punish_of_money"] for r in buffer],
            [r["criminals"] for r in buffer],
            [r["imprisonment"] for r in buffer],
            [r["life_imprisonment"] for r in buffer],
            [r["death_penalty"] for r in buffer],
            embeddings["sparse"],
            embeddings["dense"],
        ]
        col.insert(to_insert)
        total += len(buffer)
    logger.info(f"Inserted {total} records.")


def main():
    jsonl_files = []

    # 如果是目录，递归处理目录及子目录下所有的json文件
    for root, _, files in os.walk("/workspace/criminal-cases"):
        for file in files:
            if file.endswith('.json'):
                jsonl_files.append(os.path.join(root, file))

    if not jsonl_files:
        logger.error(
            f"No JSON files found in directory or subdirectories: /workspace/criminal-cases"
        )
        return
    logger.info(f"Found {len(jsonl_files)} JSON files to process")

    # Initialize clients
    logger.info("Initializing clients...")
    milvus_client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
    logger.info("Clients initialized successfully")

    # Create database
    logger.info(f"Setting up database {DATABASE_NAME}...")
    create_database(milvus_client, DATABASE_NAME)

    ef = BGEM3EmbeddingFunction(use_fp16=False, device="gcu")
    logger.info("Using Enflame GCU for embedding")
    dense_dim = ef.dim["dense"]
    col = setup_milvus_collection(dense_dim)

    # 处理每个文件
    for jsonl_path in jsonl_files:
        logger.info(f"Processing file: {jsonl_path}")
        insert_data_streaming(col, record_generator(jsonl_path), ef)

    logger.info("All files processed successfully")


if __name__ == "__main__":
    main()
