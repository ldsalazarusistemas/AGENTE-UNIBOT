FROM python:3.12-slim

WORKDIR /app

ENV KERAS_BACKEND=tensorflow \
    CUDA_VISIBLE_DEVICES=-1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p conocimiento data modelos web

EXPOSE 8000

# Un solo worker: el índice y los modelos viven en memoria del proceso.
CMD ["uvicorn", "servidor:api", "--host", "0.0.0.0", "--port", "8000"]