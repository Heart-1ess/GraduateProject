from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

import os

ROOT_DIR = "/home/zhangshuhao/projects/ys/Graduate"

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
 
 
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Tasks:
# emoji, emotion, hate, irony, offensive, sentiment
# stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary

model_path = os.path.join(ROOT_DIR, "model/emotion_roberta/twitter-roberta-base-sentiment")

tokenizer = AutoTokenizer.from_pretrained(model_path)

# handle label mapping
labels=[]
mapping_file = f"model/emotion_roberta/twitter-roberta-base-sentiment/mapping.txt"
with open(mapping_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line.strip():
            label = line.strip().split('\t')[1]
            labels.append(label)

# PT
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def embed(text: str) -> str:
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    # embed the text to a vector
    return output[0][0].detach()

if __name__ == "__main__":
    text = "Good night 😊"
    print(embed(text))