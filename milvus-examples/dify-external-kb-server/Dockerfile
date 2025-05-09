FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install -U pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -c "from pymilvus.model.hybrid import BGEM3EmbeddingFunction; BGEM3EmbeddingFunction(use_fp16=False, device='cpu')"

COPY app.py .
EXPOSE 5001

ENV DEBUG_MODE=true
ENV HF_ENDPOINT=https://hf-mirror.com
CMD ["python", "app.py"]
