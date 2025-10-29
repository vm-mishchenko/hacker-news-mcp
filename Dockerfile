FROM python:3.11-slim

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies using uv
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Unbuffered mode to see print statements
ENV PYTHONUNBUFFERED=1

# Run MCP server
CMD ["python", "src/server.py"]
