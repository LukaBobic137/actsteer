from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Meta-Llama-3-8B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "2 + 3 ="
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Generating...")
outputs = model.generate(**inputs, max_new_tokens=10)

print("Output:")
print(tokenizer.decode(outputs[0]))