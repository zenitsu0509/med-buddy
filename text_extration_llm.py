import os
import pytesseract
from PIL import Image
import requests
from dotenv import load_dotenv
load_dotenv()

file = "paracetamol-tablet.jpg"

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"  
API_TOKEN = os.getenv("HF_TOKEN")

def extract_text_from_image(image_path):
    text = pytesseract.image_to_string(Image.open(image_path))
    return " ".join(text.split())
def query_huggingface_api(text):
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    payload = {"inputs": f"Extract the medicine name from the following text: '{text}'. Only provide the name."}
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return f"Error: {response.status_code} - {response.text}"
text = extract_text_from_image(file)
process_text = query_huggingface_api(text)
print(process_text)