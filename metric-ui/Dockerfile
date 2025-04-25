# Dockerfile
FROM registry.access.redhat.com/ubi9/python-311:latest

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app2.py"]
