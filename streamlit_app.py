import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
from PIL import Image
import io

# Import your existing text extraction module
from text_extration_llm import MedicineNameExtractor

# Load environment variables
load_dotenv()

# Initialize extractor
extractor = MedicineNameExtractor()

def search_medical_database(query, top_k=5):
    # Initialize embedding model
    embedding_model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
    
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
    st.title("Medical Image Search App")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a medicine image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        try:
            # Save the uploaded file temporarily
            with open("temp_uploaded_image.png", "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Extract medicine name from the image
            extracted_text = extractor.extract_medicine_name("temp_uploaded_image.png")
            
            # Display extracted text
            st.subheader("Extracted Medicine Name")
            st.write(extracted_text)
            
            # Search medical database
            results = search_medical_database(extracted_text)
            
            # Display search results
            st.subheader("Search Results")
            for match in results['matches']:
                # Create an expandable section for each result
                with st.expander(f"Match (Similarity Score: {match['score']:.3f})"):
                    # Display metadata in a more readable format
                    metadata = match['metadata']
                    
                    # Medicine Name
                    st.markdown(f"**Medicine Name:** {metadata.get('medicine_name', 'N/A')}")
                    
                    # Composition
                    st.markdown(f"**Composition:** {metadata.get('composition', 'N/A')}")
                    
                    # Uses
                    st.markdown("**Uses:**")
                    st.write(metadata.get('uses', 'N/A'))
                    
                    # Side Effects
                    st.markdown("**Side Effects:**")
                    side_effects = metadata.get('side_effects', 'N/A')
                    # Split side effects into a list for better readability
                    side_effects_list = side_effects.split()
                    st.write(", ".join(side_effects_list))
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
        
        finally:
            # Clean up temporary file
            if os.path.exists("temp_uploaded_image.png"):
                os.remove("temp_uploaded_image.png")

if __name__ == "__main__":
    main()