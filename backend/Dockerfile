FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (cached layer — only re-runs if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend code
COPY . .

# Default command (overridden by docker-compose for dev)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]