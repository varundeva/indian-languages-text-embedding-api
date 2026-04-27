# Indian Languages Text Embedding API

A high-performance REST API for generating sentence embeddings for Indian regional languages using IndicBERT v2. This service provides semantic text representations that can be used for similarity search, clustering, and other NLP tasks.

## Features

- **Multi-language Support**: Supports 13 Indian languages including Kannada, Hindi, Tamil, Telugu, Malayalam, Marathi, Bengali, Gujarati, Punjabi, Odia, Urdu, and English
- **High Performance**: Optimized for CPU usage with configurable thread allocation
- **Batch Processing**: Process multiple texts simultaneously for efficiency
- **Docker Ready**: Containerized deployment with security best practices
- **FastAPI Framework**: Modern, async API with automatic OpenAPI documentation
- **Health Monitoring**: Built-in health checks and system metrics

## Supported Languages

| Code | Language   | Code | Language   |
|------|------------|------|------------|
| kn   | Kannada    | mr   | Marathi    |
| hi   | Hindi      | bn   | Bengali    |
| ta   | Tamil      | gu   | Gujarati   |
| te   | Telugu     | pa   | Punjabi    |
| ml   | Malayalam  | or   | Odia       |
| ur   | Urdu       | en   | English    |

## API Endpoints

### Health Check
```http
GET /health
```

Returns system status, model readiness, and resource usage.

**Response:**
```json
{
  "status": "ok",
  "model": "ai4bharat/IndicBERTv2-MLM-Sam-TLM",
  "threads_used": 5,
  "total_cpu_cores": 11,
  "cpu_cap": "50%",
  "ram_total_gb": 16.0,
  "ram_available_gb": 12.5,
  "ram_used_pct": 21.8
}
```

### Single Text Embedding
```http
POST /embed
```

Generate embedding for a single text.

**Request:**
```json
{
  "text": "ಭಾರತ ವಿಶ್ವಕಪ್ ಗೆದ್ದಿತು",
  "language": "kn"
}
```

**Response:**
```json
{
  "embedding": [0.123, 0.456, ...],
  "dimensions": 768,
  "language": "kn",
  "model": "ai4bharat/IndicBERTv2-MLM-Sam-TLM"
}
```

### Batch Text Embedding
```http
POST /embed/batch
```

Generate embeddings for multiple texts.

**Request:**
```json
{
  "items": [
    {"text": "Text 1", "language": "kn"},
    {"text": "Text 2", "language": "hi"}
  ]
}
```

**Response:**
```json
{
  "embeddings": [[0.123, ...], [0.456, ...]],
  "count": 2,
  "dimensions": 768,
  "model": "ai4bharat/IndicBERTv2-MLM-Sam-TLM"
}
```

### Supported Languages List
```http
GET /languages
```

Returns list of supported languages.

**Response:**
```json
{
  "supported": [
    {"code": "kn", "name": "Kannada"},
    {"code": "hi", "name": "Hindi"},
    ...
  ]
}
```

## Quick Start

### Using Docker (Recommended)

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd indian-languages-text-embedding
   ```

2. **Build the Docker image:**
   ```bash
   docker build -t indicbert-api .
   ```
   This performs a multi-stage build that optimizes image size by:
   - Building dependencies and downloading the model in a builder stage
   - Cleaning up Python cache files (`__pycache__`)
   - Copying only the virtual environment and saved model to the runtime stage
   - Discarding build tools (gcc, python3-dev) from the final image

3. **Run the container:**
   ```bash
   docker run -d -p 8000:8000 --name indicbert indicbert-api
   ```

4. **Test the API:**
   ```bash
   curl http://localhost:8000/health
   ```

### Local Development

1. **Install Python dependencies:**
   ```bash
   pip install fastapi uvicorn transformers torch numpy psutil psycopg2-binary
   ```

2. **Download the model (optional - speeds up startup):**
   ```python
   from transformers import AutoTokenizer, AutoModel
   AutoTokenizer.from_pretrained('ai4bharat/IndicBERTv2-MLM-Sam-TLM')
   AutoModel.from_pretrained('ai4bharat/IndicBERTv2-MLM-Sam-TLM')
   ```

3. **Run the server:**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2
   ```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `ai4bharat/IndicBERTv2-MLM-Sam-TLM` | HuggingFace model name |
| `MAX_LENGTH` | `512` | Maximum token length for input texts |
| `MAX_BATCH_SIZE` | `64` | Maximum number of texts in batch requests |
| `TEXT_PREVIEW` | `500` | Character limit before tokenization (performance optimization) |
| `CPU_USAGE_CAP` | `0.5` | Fraction of CPU cores to use (0.0-1.0) |

### Model Details

- **Model**: IndicBERT v2 (Multilingual Language Model - Script Aware Multilingual)
- **Dimensions**: 768
- **Architecture**: BERT-based transformer
- **Training Data**: Multiple Indian languages with script-aware tokenization
- **Normalization**: L2-normalized embeddings for cosine similarity

## Usage Examples

### Python Client

```python
import requests

# Single embedding
response = requests.post('http://localhost:8000/embed', json={
    'text': 'भारत विश्व कप जीता',
    'language': 'hi'
})
embedding = response.json()['embedding']

# Batch embeddings
response = requests.post('http://localhost:8000/embed/batch', json={
    'items': [
        {'text': 'ಕನ್ನಡದಲ್ಲಿ ಬರೆಯಲಾಗಿದೆ', 'language': 'kn'},
        {'text': 'Written in Hindi', 'language': 'hi'}
    ]
})
embeddings = response.json()['embeddings']
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Single embedding
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "ಭಾರತ ವಿಶ್ವಕಪ್ ಗೆದ್ದಿತು", "language": "kn"}'

# Batch embedding
curl -X POST http://localhost:8000/embed/batch \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"text": "Text 1", "language": "kn"},
      {"text": "Text 2", "language": "hi"}
    ]
  }'
```

## Performance Optimization

- **CPU Threading**: Automatically limits to 50% of available cores
- **Batch Processing**: Process multiple texts in single request
- **Text Truncation**: Pre-tokenization length limit for performance
- **Memory Management**: Efficient tensor operations with gradient detachment
- **Docker Image Optimization**:
  - Multi-stage build eliminates build tools (gcc, python3-dev) from final image
  - Python `__pycache__` files cleaned up to reduce bloat
  - Model explicitly saved (no symlinks) to avoid duplication
  - CPU-only PyTorch distribution (no CUDA bloat)

## Security

- Non-root user execution in Docker
- Minimal base image (Python slim)
- CORS enabled for web applications
- Input validation and sanitization

## Monitoring

The API provides health endpoints for monitoring:

- Model loading status
- Memory usage statistics
- CPU utilization
- Thread allocation

## Development

### Project Structure

```
├── app.py              # Main FastAPI application
├── Dockerfile          # Multi-stage Docker build
│                       # - Stage 1 (builder): Creates venv, installs deps, downloads model
│                       # - Stage 2 (runtime): Copies only venv and model files
├── requirements.txt    # Python dependencies (if not using Docker)
└── README.md          # This file
```

### Docker Build Details

The Dockerfile uses a **two-stage build process**:

**Stage 1 - Builder:**
- Creates a Python virtual environment in `/opt/venv`
- Installs all dependencies including PyTorch (CPU-only)
- Downloads and caches the IndicBERT model
- Cleans up Python cache files (`__pycache__`) to save space
- Explicitly saves the model to `/app/model` to avoid symlink overhead

**Stage 2 - Runtime:**
- Starts fresh with minimal `python:3.12-slim` base image
- Copies only the virtual environment from builder
- Copies the pre-downloaded model
- Copies application code
- Runs as non-root user for security

This approach significantly reduces image size by excluding build dependencies from the final image.

### Adding New Languages

The model supports additional Indian languages. To add support:

1. Update `SUPPORTED_LANGUAGES` dictionary in `app.py`
2. Test with sample texts
3. Update documentation

## Troubleshooting

### Common Issues

1. **Model Loading Timeout**
   - Ensure stable internet connection
   - Consider pre-downloading model locally

2. **Memory Issues**
   - Reduce `MAX_BATCH_SIZE`
   - Increase `CPU_USAGE_CAP` if more cores available

3. **Port Already in Use**
   - Change port mapping: `docker run -p 8001:8000 ...`

### Logs

View container logs:
```bash
docker logs indicbert
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [AI4Bharat](https://ai4bharat.org/) for the IndicBERT model
- [Hugging Face](https://huggingface.co/) for model hosting
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework</content>
<parameter name="filePath">/Users/varundeva/Projects/indian-languages-text-embedding/README.md