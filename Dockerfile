FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AGRISHIELD_HOST=0.0.0.0 \
    AGRISHIELD_PORT=8000 \
    AGRISHIELD_DASHBOARD_ROOT=/app/dashboard_v2 \
    AGRISHIELD_REGISTRY_PATH=/app/dashboard_v2/model_registry.json \
    AGRISHIELD_AUDIT_PATH=/data/audit.jsonl

WORKDIR /app/dashboard_v2
COPY dashboard_v2/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY dashboard_v2/ ./

RUN mkdir -p /data
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('AGRISHIELD_PORT', '8000') + '/v1/health', timeout=4)"

CMD ["python", "api_server.py"]
