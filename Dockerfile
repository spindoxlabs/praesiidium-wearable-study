FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# CPU-only torch wheels: the surrogate runs on CPU, and this keeps the image small.
COPY requirements.txt .
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

COPY constants.py run.py ./
COPY pipeline/ pipeline/

# data/ and output/ are mounted at runtime, not baked into the image.
VOLUME ["/app/data", "/app/output"]

ENTRYPOINT ["python", "run.py"]
