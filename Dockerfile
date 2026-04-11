FROM python:3.11-slim

WORKDIR /app

COPY . /app/

RUN pip install --no-cache-dir fastapi uvicorn[standard] pydantic httpx openai openenv-core

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

ENV ENABLE_WEB_INTERFACE=true

CMD ["uvicorn", "bug_review_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
