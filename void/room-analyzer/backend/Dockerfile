
# 🐍 FastAPI + YOLOv11 + MiDaS Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY backend/ /app

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
