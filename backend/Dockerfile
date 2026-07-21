FROM python:3.12-slim-bookworm

WORKDIR /app

# System dependencies for OpenCV and face ML libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# MiniFASNet — install from source (not on PyPI)
RUN pip install --no-cache-dir \
    git+https://github.com/minivision-ai/Silent-Face-Anti-Spoofing || \
    echo "MiniFASNet install failed — stub liveness detector will be used"

COPY . .

# Create storage directory
RUN mkdir -p app/storage/faces

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
