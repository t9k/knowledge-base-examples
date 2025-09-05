import re
import jieba
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_text(file_path):
    """Load text from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def preprocess_text(text):
    """Preprocess the text for better RAG performance"""
    # Remove page numbers and headers/footers
    text = re.sub(r'\n比亚迪股份有限公司 20\d{2}年年度报告全文\s*', '\n', text)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # Remove extra whitespaces and normalize line breaks
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove special characters but keep Chinese punctuation
    text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\w\s.,;:?!，。；：？！、（）()《》""''\[\]\-]', '', text)
    
    return text

def segment_chinese_text(text):
    """Use jieba to segment Chinese text for better semantic understanding"""
    segments = jieba.cut(text)
    return " ".join(segments)

def chunk_text(text, chunk_size=500, chunk_overlap=100):
    """Split text into overlapping chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", "：", "；", "，", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    return chunks

def save_chunks(chunks, output_file):
    """Save chunks to JSON file with metadata"""
    documents = []
    for i, chunk in enumerate(chunks):
        documents.append({
            "id": f"chunk_{i}",
            "text": chunk,
            "metadata": {
                "source": "BYD 2023 Annual Report",
                "chunk_index": i
            }
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

def main():
    # Load the extracted text
    input_file = "extracted_text.txt"
    output_file = "processed_chunks.json"
    
    # Process the text
    raw_text = load_text(input_file)
    clean_text = preprocess_text(raw_text)
    
    # Option: Segment with jieba (uncomment if needed)
    # segmented_text = segment_chinese_text(clean_text)
    # chunks = chunk_text(segmented_text)
    
    # Create chunks (using clean text directly)
    chunks = chunk_text(clean_text)
    
    print(f"Created {len(chunks)} chunks from the document")
    print(f"Average chunk length: {sum(len(c) for c in chunks) / len(chunks)}")
    
    # Save chunks to file
    save_chunks(chunks, output_file)
    print(f"Saved processed chunks to {output_file}")
    
    # Display a sample chunk
    if chunks:
        print("\nSample chunk:")
        print(chunks[0][:200] + "...")

if __name__ == "__main__":
    main()
