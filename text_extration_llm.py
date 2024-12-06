import os
import pytesseract
from PIL import Image
import requests
from dotenv import load_dotenv

class MedicineNameExtractor:
    def __init__(self):
        load_dotenv()
        self.api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
        self.api_token = os.getenv("HF_TOKEN")
        if not self.api_token:
            raise ValueError("API token not found. Please set 'HF_TOKEN' in the environment variables.")

    def _extract_text_from_image(self, image_path):
        try:
            text = pytesseract.image_to_string(Image.open(image_path))
            return " ".join(text.split()) 
        except Exception as e:
            raise RuntimeError(f"Error extracting text from image: {e}")

    def _query_huggingface_api(self, text):
        """Queries the Hugging Face API to extract the medicine name."""
        headers = {"Authorization": f"Bearer {self.api_token}"}
        precise_prompt = f"""
        Task: Extract the EXACT medicine name from the given text.

        Guidelines:
        - Extract ONLY the medicine name, including brand name and strength
        - Look for formats like: BRAND NAME + STRENGTH
        - Examples of correct extractions:
          * From 'PARACIP-500 Tablet' -> PARACIP-500
          * From 'Alex Syrup 100ml' -> Alex Syrup
          * From 'Calpol 500mg Suspension' -> Calpol 500mg
          * From 'Amoxil 250 Capsule' -> Amoxil 250

        Strict Rules:
        - No additional words or descriptions
        - Include strength/formulation if present in the original text
        - Prioritize complete brand name with strength

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

        response = requests.post(self.api_url, headers=headers, json=payload)
        if response.status_code == 200:
            try:
                return response.json()[0]["generated_text"].strip()
            except (KeyError, IndexError):
                raise RuntimeError("Error processing response from Hugging Face API.")
        else:
            raise RuntimeError(f"API request failed with status {response.status_code}: {response.text}")

    def extract_medicine_name(self, image_path):
        """Extract the medicine name from the given image."""
        text = self._extract_text_from_image(image_path)
        return self._query_huggingface_api(text)
