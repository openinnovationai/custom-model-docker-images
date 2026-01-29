# Reranker API

## Configuration

The application relies on the following environment variable:

- `OICM_MODEL_PATH`: **(Required)** The file system path to the directory containing the model.

## API Endpoints

### 1. Health Check
Checks the operational status of the API and the loaded model.

- **Endpoint**: `/health-check`
- **Method**: `GET`
- **Example Request**:
  ```bash
  curl http://localhost:8080/health-check
  ```
- **Response Example**:
  ```json
  {
    "status": "healthy",
    "device": "cuda"
  }
  ```
- **Error (503)**: Returns if the model is not loaded.

### 2. Rerank Documents
Ranks a list of candidate documents against a query.

- **Endpoint**: `/v1/rerank`
- **Method**: `POST`
- **Example Request**:
  ```bash
  curl -X POST http://localhost:8080/v1/rerank \
    -H "Content-Type: application/json" \
    -d '{
      "query": "What is the capital of France?",
      "documents": [
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
        "The Eiffel Tower is in Paris."
      ],
      "top_k": 3
    }'
  ```
- **Response Example**:
  Returns a list of ranked results, sorted by score.
  ```json
  [
    {
      "corpus_id": 1,
      "score": 0.85,
      "text": "Document text B"
    },
    {
      "corpus_id": 0,
      "score": 0.12,
      "text": "Document text A"
    }
  ]
  ```

## Getting Help

```sh 
make 

# OR 

make help
```