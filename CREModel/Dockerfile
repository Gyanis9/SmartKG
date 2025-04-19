FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 下载预训练模型
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='bert-base-uncased')"

EXPOSE 50051 8000
CMD ["python", "-m", "app.main"]