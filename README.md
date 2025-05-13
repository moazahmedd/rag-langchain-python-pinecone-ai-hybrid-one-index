# PDF Q&A API

A FastAPI-based API for querying PDF documents using LangChain and Pinecone.

## Project Structure

```
.
├── app/
│   ├── config/
│   │   └── settings.py
│   ├── models/
│   │   └── schemas.py
│   ├── routers/
│   │   ├── document_router.py
│   │   └── query_router.py
│   ├── services/
│   │   ├── chunking_service.py
│   │   ├── document_service.py
│   │   ├── embedding_service.py
│   │   ├── llm_service.py
│   │   └── vector_store_service.py
│   └── main.py
├── data/
│   └── Think-And-Grow-Rich.pdf
├── .env
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone API key
- A PDF document to query

## Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

5. Create a data directory and place your PDF file:

```bash
mkdir data
# Copy your PDF file to the data directory
cp path/to/your/pdf data/Think-And-Grow-Rich.pdf
```

## Running the API

1. Start the FastAPI server:

```bash
cd app
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### 1. Upload Document

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload"
```

### 2. Query Document

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the main topic?", "k": 3}'
```

### 3. Delete Document

```bash
curl -X DELETE "http://localhost:8000/api/v1/documents"
```

## Testing with Postman

1. Import the following collection into Postman:

```json
{
  "info": {
    "name": "PDF Q&A API",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "Upload Document",
      "request": {
        "method": "POST",
        "url": "http://localhost:8000/api/v1/documents/upload"
      }
    },
    {
      "name": "Query Document",
      "request": {
        "method": "POST",
        "url": "http://localhost:8000/api/v1/query",
        "header": [
          {
            "key": "Content-Type",
            "value": "application/json"
          }
        ],
        "body": {
          "mode": "raw",
          "raw": "{\n    \"query\": \"What is the main topic?\",\n    \"k\": 3\n}"
        }
      }
    },
    {
      "name": "Delete Document",
      "request": {
        "method": "DELETE",
        "url": "http://localhost:8000/api/v1/documents"
      }
    }
  ]
}
```

## Common Issues

1. **PDF File Not Found**

   - Ensure the PDF file is in the `data` directory
   - Check the filename matches `Think-And-Grow-Rich.pdf`
   - Verify the data directory exists in the project root

2. **API Key Issues**

   - Verify your API keys in the .env file
   - Ensure the keys are valid and have proper permissions

3. **Pinecone Index Issues**
   - Make sure the index exists in your Pinecone account
   - Check if the namespace is properly configured

## Development

To run tests (when implemented):

```bash
pytest
```

To check code style:

```bash
flake8
```
