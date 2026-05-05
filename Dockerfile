FROM python:3.12-slim

WORKDIR /app

# Install system deps for PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e "."

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Runtime env
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV WORKSPACE=/app/data/workspace
ENV DB_PATH=/app/data/index.db

# Data volume for persistent storage
VOLUME ["/app/data"]

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -fsS http://localhost:3000/health || exit 1

CMD ["python", "server.py"]
