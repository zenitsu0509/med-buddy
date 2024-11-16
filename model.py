import torch
from transformers import AutoTokenizer, BertForQuestionAnswering

# Define model path
model_path = r"E:\GIT REPOS\medince project\med-buddy\model"

# Load tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")

# Load model
try:
    model = BertForQuestionAnswering.from_pretrained(model_path, local_files_only=True)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Test with a sample question
question = "What are the uses of Avastin 400mg Injection?"
context = (
    "Medicine Name: Avastin 400mg Injection, Composition: Bevacizumab (400mg), "
    "Uses: Cancer of colon and rectum, lung cancer, "
    "Side Effects: Headache, taste change, rectal bleeding."
)

# Tokenize input
inputs = tokenizer(question, context, return_tensors="pt", truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Get model predictions
with torch.no_grad():
    outputs = model(**inputs)

# Extract start and end logits
start_idx = torch.argmax(outputs.start_logits)
end_idx = torch.argmax(outputs.end_logits) + 1

# Decode the predicted answer
answer = tokenizer.decode(inputs['input_ids'][0][start_idx:end_idx])
print(f"Predicted Answer: {answer}")
