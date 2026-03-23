import pandas as pd
import os

def load_claims(file_path):
    """Loads the CSV and returns claims with essential columns."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
        
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Keep the identifiers and description for embedding
    required_cols = ['ClaimNumber', 'ClaimDescription']
    
    # Basic validation: ensure columns exist
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from the dataset.")
    
    return df[required_cols]

if __name__ == "__main__":
    # Test
    try:
        data = load_claims("data/insurance_claims.csv")
        print(f"Successfully loaded {len(data)} claims.")
        print(data.head())
    except Exception as e:
        print(f"Error: {e}")
