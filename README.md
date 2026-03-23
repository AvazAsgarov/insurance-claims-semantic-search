![Insurance Claims Project Banner](file:///c:/Users/avaza/Desktop/pin/assests/banner.png)

# Insurance Claim Processing with Pinecone

This project demonstrates a functional implementation for analyzing workers' compensation claims through semantic vector search. By utilizing OpenAI embeddings and a Pinecone vector database, the system identifies similar safety incidents and historical claim patterns.

## Project Structure

The repository is organized following modular principles to ensure clear separation of concerns:

```text
pin/
- `data/`: Raw workers' compensation dataset
- `src/config.py`: Environment and global configuration
- `src/data_loader.py`: Dataset ingestion utilities
- `src/embeddings.py`: OpenAI vectorization service
- `src/vector_db.py`: Pinecone storage and query logic
- `src/main.py`: Orchestration and analysis entry point
├── .env                  # API keys (exclude from version control)
└── requirements.txt      # Python dependencies
```

## Implementation Workflow

The implementation follows a modular, functional architecture:

1.  **Configuration**: Secure extraction of API keys for OpenAI and Pinecone from protected environment files.
2.  **Ingestion**: Loading and validation of claim identifiers and textual descriptions via Pandas.
3.  **Vectorization**: Generation of 1536-dimensional semantic representations using the OpenAI text-embedding-3-small model.
4.  **Indexing**: Initialization of a Pinecone serverless index and batch upsertion of vectorized records.
5.  **Retrieval**: Transformation of natural language safety queries into vectors for high-precision semantic matching.

## Analysis Results

The system successfully identified the most semantically relevant claims for the specified safety scenarios using cosine similarity.

| Target Scenario | Closest Claim ID | Matched Claim Description |
| :--- | :--- | :--- |
| **Rear-end Collision** | WC8133442 | COLLISION WITH MOTOR VEHICLE ACCIDENT SORE NECK |
| **Carpal Tunnel / Repetitive Strain** | WC8450357 | WHILE DEALING CARDS RIGHT TENDON SYNOVITIS RIGHT WRIST |

## Setup and Execution

1. Install the required libraries:

```bash
pip install -r requirements.txt
```

2. Configure the `.env` file with functional OPENAI_API_KEY and PINECONE_API_KEY.

3. Execute the main analysis module:

```bash
python -m src.main
```
