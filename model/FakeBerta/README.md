---
license: apache-2.0
language:
- en
metrics:
- accuracy: 1
- recall: 1
- f1: 1
base_model:
- distilbert/distilroberta-base
tags:
- fake-news
- text-classification
- transformers
- distilroberta
- nlp
- deep-learning
- pytorch
- huggingface
- fine-tuning
- misinformation
---
**FakeBerta: A Fine-Tuned DistilRoBERTa Model for Fake News Detection**

You can check the model's fine-tuning code on my GitHub.

Model Overview

FakeBerta is a fine-tuned version of DistilRoBERTa for detecting fake news. The model is trained to classify news articles as real (0) or fake (1) using natural language processing (NLP) techniques.
Base Model: DistilRoBERTa
Task: Fake news classification

Example of code using AutoModelForSequenceCalssification:
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_name = "YerayEsp/FakeBerta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

inputs = tokenizer("Breaking: Scientists discover water on Mars!", return_tensors="pt")
outputs = model(**inputs)

logits = outputs.logits
predicted_class = torch.argmax(logits).item()

print(f"Predicted class: {predicted_class}")  # 0 = Real, 1 = Fake

```
Library: Transformers (Hugging Face)