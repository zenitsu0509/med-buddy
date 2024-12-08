import os
import pytesseract
from PIL import Image
import requests
from dotenv import load_dotenv


class MedicineNameExtractor:
    def __init__(self):
        """Initialize the Medicine Name Extractor with Hugging Face API credentials."""
        load_dotenv()
        self.api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
        self.api_token = os.getenv("HF_TOKEN")
        if not self.api_token:
            raise ValueError("API token not found. Please set 'HF_TOKEN' in the environment variables.")

    def _extract_text_from_image(self, image_path):
        """Extract text from the given image using Tesseract OCR."""
        try:
            text = pytesseract.image_to_string(Image.open(image_path))
            return " ".join(text.split())  # Clean up whitespace
        except Exception as e:
            raise RuntimeError(f"Error extracting text from image: {e}")

    def _query_huggingface_api(self, text):
        """Query the Hugging Face API to extract the brand name from the text."""
        headers = {"Authorization": f"Bearer {self.api_token}"}
        precise_prompt = f"""
        You are a medical drug store operator, who is expert in medicines and names. 
        As an expert, your task is to extract the brand name of the medicine based on the input provided.
        Always return the response in the following format:

        Example: 
        {{'input': 'PARACIP-500 Tablet', 'output': 'PARACIP'}}
        {{'input': 'Azithromycin Tablets IP 500 mg Azithral-500', 'output': 'Azithral'}}

        Text to process: '{text}'

        ONLY return the exact medicine name or the most precise identifier.
        """
        payload = {
            "inputs": precise_prompt,
            "parameters": {
                "max_new_tokens": 50,
                "temperature": 0.1,
                "repetition_penalty": 1.1
            }
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            if response.status_code == 200:
                try:
                    return response.json()[0]["generated_text"].strip()
                except (KeyError, IndexError):
                    raise RuntimeError("Unexpected response structure from Hugging Face API.")
            else:
                raise RuntimeError(f"API request failed with status {response.status_code}: {response.text}")
        except requests.RequestException as e:
            raise RuntimeError(f"Error querying Hugging Face API: {e}")

    def extract_medicine_name(self, image_path):
        """Extract the medicine name from the given image."""
        text = self._extract_text_from_image(image_path)
        return self._query_huggingface_api(text)
