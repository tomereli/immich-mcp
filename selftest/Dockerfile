FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir "httpx>=0.27"
COPY probe.py .
CMD ["python", "probe.py"]
