FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train so the model artifact exists in the image.
RUN python train_model.py

EXPOSE 8000

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
