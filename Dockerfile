# On-premise image: everything the OCR pipeline needs is baked in, because
# the deployment target may have no route to the internet at all.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # PaddleX's oneDNN backend crashes during text detection on some CPU
    # builds. Must be set before paddle is imported anywhere in the process
    # — including the warm-up below, which would otherwise fail at build time.
    PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=0 \
    # Where PaddleX caches downloaded model weights. This exact name matters:
    # paddlex reads PADDLE_PDX_CACHE_HOME (see paddlex/utils/cache.py) and
    # ignores anything else, so a plausible-looking PADDLEX_HOME would send
    # the build's download to ~/.paddlex while the runtime looked elsewhere.
    PADDLE_PDX_CACHE_HOME=/opt/paddlex \
    # Skip the startup connectivity check to the model hosters. Without this
    # every worker process reaching OCR first tries to contact huggingface.co
    # (and modelscope, aistudio, bos) to see which are reachable — on an
    # isolated network that is a series of TCP timeouts before any invoice is
    # read. The weights are already local, so there is nothing to check.
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# Runtime libraries the vision stack links against. opencv-python-headless
# avoids most of the X11 chain, but PaddleX still needs libgomp for its
# OpenMP kernels and libglib for its image codecs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 180 --retries 10 -r requirements.txt

# Download the *paddle* OCR model weights at BUILD time, so the first request
# on an air-gapped box does not try to fetch them and fail. Without this the
# image looks fine in CI (which has a network) and dies on the customer's
# machine.
#
# The default backend is ONNX Runtime, whose models ship inside the rapidocr
# wheel and are therefore already present after the pip install above. This
# step is for the paddle fallback backend (OCR_BACKEND=paddle), which is the
# one that fetches its weights at first use.
#
# The connectivity check is re-enabled for this one command: the build *does*
# have a network, and letting paddlex pick a reachable hoster is more robust
# than pinning one that might be blocked from the build machine.
RUN PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK= python -c "\
from paddleocr import PaddleOCR; \
PaddleOCR(lang='en', use_doc_orientation_classify=False, \
          use_doc_unwarping=False, use_textline_orientation=False)"

COPY . .

# Fail the build if the weights did not land, rather than shipping an image
# that looks fine until a customer uploads their first invoice. Checks the
# models directory specifically — the cache root also holds lock and temp
# directories, which exist even when nothing was downloaded.
RUN python - <<'PYTHON'
import os
import sys
from pathlib import Path

# The default backend. rapidocr ships these inside its wheel, so they are
# present or the install itself was broken — but check anyway, because the
# failure mode being guarded against is precisely one that stays invisible
# until a customer uploads an invoice.
onnx_models = Path(__import__("rapidocr").__file__).parent / "models"
onnx_present = sorted(path.name for path in onnx_models.glob("*.onnx"))
if not onnx_present:
    sys.exit(f"ONNX OCR models are missing from {onnx_models}; the image "
             "would need network access at run time.")
print(f"ONNX OCR models bundled: {onnx_present}")

# The fallback backend, fetched by the build step above.
paddle_models = os.path.join(os.environ["PADDLE_PDX_CACHE_HOME"], "official_models")
if not os.path.isdir(paddle_models) or not any(os.scandir(paddle_models)):
    sys.exit(f"Paddle OCR model weights are missing from {paddle_models}; the "
             "image would need network access at run time.")
print(f"Paddle OCR models cached: {sorted(entry.name for entry in os.scandir(paddle_models))}")
PYTHON

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
