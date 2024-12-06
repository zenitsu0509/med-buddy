import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pinecone
import uuid
import os
from dotenv import load_dotenv
from text_extration_llm import MedicineNameExtractor

# Load environment variables
load_dotenv()

# Initialize extractor
extractor = MedicineNameExtractor()

def search_medical_database(query, top_k=5):
    # Initialize embedding model
    embedding_model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
    
    # Correct Pinecone initialization for latest version
    from pinecone import Pinecone, ServerlessSpec
    
    # Initialize Pinecone client
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # Connect to the index
    index = pc.Index("medical-database")
    
    # Generate query embedding
    query_embedding = embedding_model.encode(query).tolist()

    # Perform search
    results = index.query(
        vector=query_embedding, 
        top_k=top_k, 
        include_metadata=True
    )
    
    return results

def main():
    path = "images/azithral-500.jpg"

    try:
        # Extract medicine name
        text = extractor.extract_medicine_name(path)
        print(f"Extracted Medicine Name: {text}")
        
        # Search medical database
        results = search_medical_database(text)
        
        # Print results
        print("\nSearch Results:")
        for match in results['matches']:
            print("Metadata:", match['metadata'])
            print("Score:", match['score'])
            print("---")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the main function
if __name__ == "__main__":
    main()