FROM python:3.13-slim

# libgomp нужен lightgbm / catboost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# зависимости отдельным слоем — кэшируется между сборками
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# код пакета
COPY uplift/ ./uplift/
COPY README.md .env.example ./

# self-check: образ собрался и пакет импортируется
CMD ["python", "-c", "import uplift; print('autoprep-uplift image OK')"]