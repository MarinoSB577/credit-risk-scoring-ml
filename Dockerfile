FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/api/ .
COPY dashboard/models/ ./dashboard/models/

# Ruta del modelo dentro del contenedor
ENV MODEL_PATH=/app/dashboard/models/model.pkl

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]