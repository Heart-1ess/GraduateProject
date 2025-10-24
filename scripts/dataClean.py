'''

This script will serve the purpose of cleaning data for model training.

'''

import re
import nltk
from nltk.corpus import stopwords

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9 ]", "", text)
    text = text.lower()
    return text

# df['cleaned_comment'] = df['comment'].apply(clean_text)