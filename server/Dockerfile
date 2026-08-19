FROM python:3.12-slim

# ffmpeg carries ffprobe with it. Both are needed for video frame extraction:
# Immich reports a duration and nothing else for a video - no fps, no rotation,
# no audio flag - and only a decoder can seek to a chosen moment and tell you
# the timestamp it actually landed on.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY immich_mcp.py .

EXPOSE 3000
CMD ["python", "immich_mcp.py"]
