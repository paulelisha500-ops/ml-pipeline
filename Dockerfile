FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

ENV PYTHONPATH=/app/src

# No arguments prints the model's metrics. To score a file:
#   docker run -v "$PWD":/work IMAGE python src/predict.py /work/in.csv /work/out.csv
CMD ["python", "src/predict.py"]
