from unsloth import FastLanguageModel
from transformers import AutoTokenizer
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.cuda.is_available = lambda: False
torch.cuda.get_device_capability = lambda device=None: (0, 0)
model_dir = "model"  

tokenizer = AutoTokenizer.from_pretrained(model_dir)

model = FastLanguageModel.from_pretrained(
    model_name=model_dir,
    adapter_path=f"{model_dir}/adapter_model.safetensors",
    max_seq_length=5020,
    load_in_4bit=False, 
    dtype=None,
)

model.eval()

def test_model(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    device = torch.device("cpu")
    model = model.to(device)
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_length=512,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

test_input = "Tamdura Capsule PR"
result = test_model(test_input)
print("Input:", test_input)
print("Generated Output:", result)
