from pinecone import Pinecone, ServerlessSpec
import src.config as config
import time

def get_pinecone_client():
    """Returns a connected Pinecone client."""
    config.validate_config()
    return Pinecone(api_key=config.PINECONE_API_KEY)

def initialize_index(pc):
    """Ensures the index exists and is ready for use."""
    index_name = config.PINECONE_INDEX_NAME
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"Creating Pinecone index: {index_name}...")
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        print("Index initialized successfully.")
    else:
        print(f"Index '{index_name}' already exists.")
    
    return pc.Index(index_name)

def upsert_claims(index, claim_ids, vectors, descriptions):
    """Upserts claim data into Pinecone."""
    records = []
    for cid, vec, desc in zip(claim_ids, vectors, descriptions):
        records.append({
            "id": cid,
            "values": vec,
            "metadata": {"ClaimDescription": desc}
        })
    index.upsert(vectors=records)
    print(f"Successfully upserted {len(records)} records.")

def query_similar_claims(index, query_vector, top_k=1):
    """Queries the index for the most similar claims."""
    return index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

if __name__ == "__main__":
    # Test
    try:
        pc = get_pinecone_client()
        initialize_index(pc)
        print("Pinecone connection verified.")
    except Exception as e:
        print(f"Error: {e}")
