from openai import OpenAI
import src.config as config

def get_embeddings(text_list):
    """Generates embeddings for a list of strings using OpenAI."""
    if not text_list:
        return []
        
    # Initialize client locally or pass it as an argument
    config.validate_config()
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    # Clean inputs
    text_list = [text.replace("\n", " ") for text in text_list]
    
    response = client.embeddings.create(
        input=text_list,
        model=config.EMBEDDING_MODEL
    )
    
    return [item.embedding for item in response.data]

def get_embedding(text):
    """Generates embedding for a single query string."""
    return get_embeddings([text])[0]

if __name__ == "__main__":
    # Test
    try:
        test_text = "Car accident with rear-end collision"
        vector = get_embedding(test_text)
        print(f"Successfully generated embedding. Vector length: {len(vector)}")
    except Exception as e:
        print(f"Error: {e}")
