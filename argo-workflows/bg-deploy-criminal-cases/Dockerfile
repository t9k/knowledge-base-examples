FROM registry.qy.t9kcloud.cn/topsrider/jupyterlab-torch-2.3.0:20241224

ENV HF_ENDPOINT=https://hf-mirror.com

RUN pip install -U pip && \
  pip install --no-cache-dir pymilvus[model]==2.5.7 langchain_text_splitters==0.3.8 && \
  # 预下载 bge-m3 模型
  python -c "from pymilvus.model.hybrid import BGEM3EmbeddingFunction; BGEM3EmbeddingFunction(use_fp16=False, device='cpu')"
