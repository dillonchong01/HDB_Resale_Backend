# Start from a lightweight Python image
FROM python:3.10-slim

# Update apt repository to a more reliable mirror (e.g., US mirror)
RUN echo "deb http://ftp.us.debian.org/debian/ bookworm main" > /etc/apt/sources.list

# Install required system dependencies
RUN apt-get update --allow-insecure-repositories && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
    
# Set a working directory in the container
WORKDIR /app

# Copy dependency manifest and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your code into the container
COPY . .

# Expose the port your FastAPI app uses
EXPOSE 8000

# Tell Docker how to start your app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]