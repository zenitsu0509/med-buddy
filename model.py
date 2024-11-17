from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_path = "./med-buddy-llm"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

def generate_medicine_info(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_length=100,
            num_beams=4,
            early_stopping=True
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

test_input = "Paracetamol Tablets PARA 500"
output = generate_medicine_info(test_input)
print(f"Input: {test_input}\nOutput: {output}")
