FROM registry.qy.t9kcloud.cn/topsrider/jupyterlab-torch-2.3.0:20241224

ENV HF_ENDPOINT=https://hf-mirror.com

WORKDIR /app
RUN pip install -U pip && \
    pip install --no-cache-dir datasets==3.6.0 flask==2.3.3 peft==0.15.2 pymilvus[model]==2.5.7 requests==2.32.3 werkzeug==3.1.3 && \
    python -c "from pymilvus.model.hybrid import BGEM3EmbeddingFunction; BGEM3EmbeddingFunction(use_fp16=False, device='cpu')"

COPY app.py config.py ./
EXPOSE 5001

CMD ["python", "app.py"]
