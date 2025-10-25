FROM python:3.11-slim

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies using uv
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Set up Python environment
ENV PYTHONUNBUFFERED=1

# Command to run the server (unbuffered mode to see print statements)
CMD ["python", "src/server.py"]
