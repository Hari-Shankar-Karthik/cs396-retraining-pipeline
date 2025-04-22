from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

inputs = tokenizer("def add(x, y): return x + y", return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
print(outputs.logits)
