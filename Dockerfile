FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir fastapi uvicorn[standard] pydantic httpx openai openenv-core

# Copy the full project
COPY . /app/

# Expose HF Spaces port (7860)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Start the FastAPI server
CMD ["uvicorn", "bug_review_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
