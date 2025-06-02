FROM registry.access.redhat.com/ubi9/python-311:latest
WORKDIR /app
COPY mcp.py .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["uvicorn", "mcp:app", "--host", "0.0.0.0", "--port", "8000"]