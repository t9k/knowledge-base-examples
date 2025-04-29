import json
from tqdm import tqdm
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 配置
MILVUS_URI = "http://app-milvus-xxxxxxxx.namespace.svc.cluster.local:19530"
COLLECTION_NAME = "law_hybrid_demo"
CHUNK_SIZE = 256
CHUNK_OVERLAP = 32
BATCH_SIZE = 50

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
        separators=["。", "，"],
        keep_separator="end"
    )
    return text_splitter.split_text(text)

def setup_milvus_collection(dense_dim):
    connections.connect(uri=MILVUS_URI, token="root:Milvus")
    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
        FieldSchema(name="fact", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="relevant_articles", dtype=DataType.INT64),
        FieldSchema(name="accusation", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="punish_of_money", dtype=DataType.INT64),
        FieldSchema(name="criminals", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="imprisonment", dtype=DataType.INT64),
        FieldSchema(name="life_imprisonment", dtype=DataType.BOOL),
        FieldSchema(name="death_penalty", dtype=DataType.BOOL),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
    ]
    schema = CollectionSchema(fields)
    if utility.has_collection(COLLECTION_NAME):
        Collection(COLLECTION_NAME).drop()
    col = Collection(COLLECTION_NAME, schema, consistency_level="Strong")
    sparse_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
    col.create_index("sparse_vector", sparse_index)
    dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
    col.create_index("dense_vector", dense_index)
    col.load()
    return col

def record_generator(jsonl_path):
    for item in read_jsonl(jsonl_path):
        fact = item["fact"]
        meta = item["meta"]
        chunks = chunk_text(fact) if len(fact) > CHUNK_SIZE else [fact]
        for chunk in chunks:
            yield {
                "fact": chunk,
                "relevant_articles": meta.get("relevant_articles", [None])[0],
                "accusation": meta.get("accusation", [None])[0],
                "punish_of_money": meta.get("punish_of_money", None),
                "criminals": meta.get("criminals", [None])[0],
                "imprisonment": meta.get("term_of_imprisonment", {}).get("imprisonment", None),
                "life_imprisonment": meta.get("term_of_imprisonment", {}).get("life_imprisonment", False),
                "death_penalty": meta.get("term_of_imprisonment", {}).get("death_penalty", False),
            }

def insert_data_streaming(col, record_iter, ef, batch_size=BATCH_SIZE):
    buffer = []
    total = 0
    for record in tqdm(record_iter, desc="Inserting records"):
        buffer.append(record)
        if len(buffer) >= batch_size:
            facts = [r["fact"] for r in buffer]
            embeddings = ef(facts)
            to_insert = [
                facts,
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
        facts = [r["fact"] for r in buffer]
        embeddings = ef(facts)
        to_insert = [
            facts,
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
    print(f"Inserted {total} records.")

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python insert_data.py <jsonl_file>")
        return
    jsonl_path = sys.argv[1]
    ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
    dense_dim = ef.dim["dense"]
    col = setup_milvus_collection(dense_dim)
    insert_data_streaming(col, record_generator(jsonl_path), ef)

if __name__ == "__main__":
    main()
