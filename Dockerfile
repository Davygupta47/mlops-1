FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir \
	dvc>=3.67.1 \
	fastapi>=0.136.0 \
	joblib>=1.5.3 \
	mlflow>=3.11.1 \
	numpy>=2.4.4 \
	pandas>=2.3.3 \
	polars>=1.40.0 \
	scikit-learn>=1.8.0 \
	uvicorn>=0.44.0

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]