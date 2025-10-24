from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax

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

# download label mapping
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
model.save_pretrained(model_path)

text = "Good night 😊"
text = preprocess(text)
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
scores = output[0][0].detach().numpy()
scores = softmax(scores)

ranking = np.argsort(scores)
ranking = ranking[::-1]
for i in range(scores.shape[0]):
    l = labels[ranking[i]]
    s = scores[ranking[i]]
    print(f"{i+1}) {l} {np.round(float(s), 4)}")
