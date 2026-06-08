# Use an official Python 3.11 slim image (Bookworm) for stability
FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
# WeasyPrint requires: libcairo2, libpango-1.0-0, libpangocairo-1.0-0, libgdk-pixbuf-2.0-0, shared-mime-info
# Build dependencies for uvloop/hiredis: build-essential, libffi-dev
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    shared-mime-info \
    fonts-liberation \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install WeasyPrint explicitly (ensuring it matches the version in pyproject.toml)
RUN pip install --no-cache-dir "weasyprint>=60.0"

# Copy the rest of the application code
COPY . .

# Create the outputs directory and ensure it has correct permissions
RUN mkdir -p outputs && chmod 777 outputs

# Expose the port the FastAPI app runs on
EXPOSE 8000

# Healthcheck to ensure the container is running correctly
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start the application using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
