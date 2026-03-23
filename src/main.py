import src.data_loader as data_loader
import src.embeddings as embeddings
import src.vector_db as vector_db

def run_analysis():
    """Main execution flow without classes."""
    
    # 1. Initialization
    print("Initializing project services...")
    pc = vector_db.get_pinecone_client()
    index = vector_db.initialize_index(pc)
    
    # 2. Data Loading
    print("Loading insurance claims dataset...")
    df = data_loader.load_claims("data/workers_comp.csv")
    
    # 3. Embedding Generation
    print(f"Generating embeddings for {len(df)} descriptions...")
    descriptions = df['ClaimDescription'].tolist()
    vectors = embeddings.get_embeddings(descriptions)
    
    # 4. Pinecone Upsertion
    print("Upserting vectors to Pinecone...")
    vector_db.upsert_claims(
        index=index,
        claim_ids=df['ClaimNumber'].tolist(),
        vectors=vectors,
        descriptions=descriptions
    )
    
    # 5. Queries
    print("\n--- Project Findings ---")
    
    # Scenario 1: Rear-end Collision
    query_1 = "Car accident with rear-end collision"
    print(f"Querying for: '{query_1}'...")
    vec_1 = embeddings.get_embedding(query_1)
    res_1 = vector_db.query_similar_claims(index, vec_1, top_k=1)
    
    if res_1.matches:
        match = res_1.matches[0]
        print(f"1. Closest Claim ID: {match.id}")
        print(f"2. Closest Claim Description: {match.metadata['ClaimDescription']}")
    
    # Scenario 2: Carpal Tunnel
    query_2 = "Worker developed carpal tunnel syndrome from repetitive typing"
    print(f"\nQuerying for: '{query_2}'...")
    vec_2 = embeddings.get_embedding(query_2)
    res_2 = vector_db.query_similar_claims(index, vec_2, top_k=1)
    
    if res_2.matches:
        match = res_2.matches[0]
        print(f"3. Carpal Tunnel Case Description: {match.metadata['ClaimDescription']}")

if __name__ == "__main__":
    try:
        run_analysis()
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
