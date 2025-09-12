FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

ENV HF_ENDPOINT=https://hf-mirror.com

RUN pip install --no-cache-dir --upgrade pip && \
pip install --no-cache-dir "pymilvus[model]"==2.5.10 langchain_text_splitters==0.3.8 openai==1.92.2 && \
python -c "from pymilvus.model.hybrid import BGEM3EmbeddingFunction; BGEM3EmbeddingFunction(use_fp16=False, device='cpu')"
