# Use a Python image that's suitable for installing system packages
FROM python:3.10

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-rus \
    python3-opencv \
    poppler-utils \
    unoconv \
    libreoffice \
    libreoffice-writer \
    default-jre \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for LibreOffice
ENV HOME=/tmp \
    PYTHONPATH=/usr/lib/python3/dist-packages

# Set the working directory within the container
WORKDIR /app

# Copy requirements.txt before other files to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install -U --no-cache-dir -r requirements.txt
RUN pip install -U --no-cache-dir numpy==1.26.4
RUN pip install python-multipart

# Copy the source code
COPY . .

# Ensure all users can write to tmp directory
RUN chmod 1777 /tmp

# Expose the FastAPI port
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "rest:app", "--host", "0.0.0.0", "--port", "8000"]