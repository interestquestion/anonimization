# Use a Python image that's suitable for installing system packages
FROM python:3.10

# Install system dependencies, including tesseract
RUN apt-get update && \
    apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-rus \
    python3-opencv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory within the container
WORKDIR /app

# Copy requirements.txt before other files to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install -U --no-cache-dir -r requirements.txt
RUN pip install -U --no-cache-dir numpy==1.26.4

# Copy the source code
COPY . .

# Expose the FastAPI port (adjust if needed)
EXPOSE 8000

# Run the FastAPI application 
CMD ["uvicorn", "rest:app", "--host", "0.0.0.0", "--port", "8000"]