# Use a Python base image (FastAPI + ONNX needs at least 3.8)
FROM python:3.10-slim

# Create a non-root user (required by Hugging Face)
RUN useradd -m -u 1000 user
USER user

# Add pip user directory to PATH
ENV PATH="/home/user/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy dependencies file and install
COPY --chown=user requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY --chown=user . /app

# Expose default Hugging Face port
EXPOSE 7860

# Run FastAPI app using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
