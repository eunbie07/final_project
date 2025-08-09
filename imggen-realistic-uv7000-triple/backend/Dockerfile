FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml /app/
RUN pip install uv && uv pip install --system -e .
COPY . /app
EXPOSE 7000
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7000"]
