FROM python:3.11-slim
WORKDIR /app

# 替换 apt 源为清华镜像
RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt .
# 先装 CPU-only torch（避免拉几百MB的CUDA依赖）
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
# 再装其他依赖
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
COPY . .
RUN mkdir -p /app/data
EXPOSE 8000 8089
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
