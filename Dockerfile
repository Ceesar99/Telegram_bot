FROM python:3.10-slim

# System deps for TA-Lib and scientific stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    wget \
    git \
    libffi-dev \
    libssl-dev \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source
COPY . /app

# Install TA-Lib from source tarball if present, otherwise via pip prebuilt wheel
RUN if [ -f "/app/ta-lib-0.4.0-src.tar.gz" ]; then \
      mkdir -p /tmp/ta-lib && tar -xzf /app/ta-lib-0.4.0-src.tar.gz -C /tmp/ta-lib --strip-components=1 && \
      cd /tmp/ta-lib && ./configure --prefix=/usr && make && make install && \
      rm -rf /tmp/ta-lib; \
    fi

# Upgrade pip and install python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir python-dotenv && \
    pip install --no-cache-dir git+https://github.com/MrAlpert/Pocket-Option-API.git

# Create runtime dirs
RUN mkdir -p /workspace/logs /workspace/data /workspace/models /workspace/backup

# Set environment
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Default command runs the universal launcher in production mode
CMD ["python", "universal_trading_launcher.py", "--mode", "ultimate", "--deployment", "production"]

