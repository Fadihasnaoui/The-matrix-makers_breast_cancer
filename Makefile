.PHONY: help install train run clean metrics

help:
	@echo "Breast Cancer Detection Dashboard"
	@echo ""
	@echo "Available commands:"
	@echo "  make install - Install dependencies"
	@echo "  make train   - Train the ML model"
	@echo "  make run     - Start FastAPI server"
	@echo "  make metrics - View available metrics"
	@echo "  make clean   - Clean up generated files"

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

train:
	@echo "Training model..."
	python scripts/train_model.py
	@echo "✅ Model training complete"

run:
	@echo "Starting API server..."
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

metrics:
	@echo "📊 Available metrics at: http://localhost:8000/metrics"
	@echo "📈 Health check: http://localhost:8000/health"
	@echo "📋 Model info: http://localhost:8000/model/info"
	@echo "🔮 Make prediction: POST http://localhost:8000/predict"
	@echo "🔄 Retrain model: POST http://localhost:8000/retrain"

clean:
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "✅ Cleanup complete"
