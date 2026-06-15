FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYNGUIN_DANGER_AWARE=1 RUNNER_WORKSPACE=/workspace \
    PYNGUIN_SOURCE_ROOT=/opt/pynguin PYTHONPATH=/opt/pynguin/src

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/pynguin /opt/pynguin
RUN pip install --no-cache-dir -e "/opt/pynguin[openai]"

COPY run_pynguin.py .

RUN mkdir -p /workspace

VOLUME ["/workspace"]

ENTRYPOINT ["python", "/app/run_pynguin.py"]
