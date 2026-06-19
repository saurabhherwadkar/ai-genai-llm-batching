# Dockerfile
# Multi-stage build for the LLM Batching application.
# Stage 1: Install dependencies with Poetry.
# Stage 2: Copy only runtime files for a minimal production image.

# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

# Set working directory for the build stage
WORKDIR /app

# Install Poetry for dependency resolution
RUN pip install --no-cache-dir poetry==2.1.1

# Copy dependency specification files first for Docker layer caching
COPY pyproject.toml poetry.lock ./

# Export dependencies to requirements.txt for pip install in production stage
RUN poetry export -f requirements.txt --output requirements.txt --without=dev

# Stage 2: Production runtime image
FROM python:3.12-slim AS runtime

# Set working directory for the application
WORKDIR /app

# Create non-root user for security (principle of least privilege)
RUN useradd --create-home --shell /bin/bash appuser

# Copy requirements from builder stage
COPY --from=builder /app/requirements.txt ./

# Install runtime dependencies only
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY llm_batching/ ./llm_batching/
COPY config/ ./config/

# Switch to non-root user for runtime
USER appuser

# Set environment variables for Python runtime behavior
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command to run the batch processing demonstration
CMD ["python", "-m", "llm_batching.main"]
