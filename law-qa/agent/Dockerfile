FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# 基础依赖（尽量精简）
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

# 先仅安装运行所需依赖，减少镜像层缓存失效
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
      "transformers>=4.55.0" \
      "fastapi>=0.110.0" \
      "uvicorn>=0.24.0" \
      "python-dateutil>=2.9.0.post0" \
      "dotenv>=0.9.9"

# 从源码安装 Qwen-Agent（带 extras）
RUN git clone https://github.com/xyxxxxx/Qwen-Agent.git /tmp/Qwen-Agent && \
    cd /tmp/Qwen-Agent && \
    pip install --no-cache-dir '.[gui,rag,code_interpreter,mcp]' && \
    rm -rf /tmp/Qwen-Agent

# 拷贝代码
COPY . /app

EXPOSE 8001

# 默认以 OpenAI 兼容 API 模式启动
CMD ["python", "app.py", "--mode", "api", "--api-host", "0.0.0.0", "--api-port", "8001"]
