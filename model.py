from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Path to the folder where the trained model is saved
model_path = "E:\\GIT REPOS\\medince project\\med-buddy\\model"

# Load the tokenizer and model from the path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)  # Safetensors support

# Function to generate medicine information
def generate_medicine_info(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_length=200,
            num_beams=4,
            early_stopping=True
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Test the model with a sample input
test_input = "Avastin 400mg Injection"
output = generate_medicine_info(test_input)
print(f"Input: {test_input}\nOutput: {output}")
