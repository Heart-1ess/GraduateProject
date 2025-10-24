import torch
import os
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = "/home/zhangshuhao/projects/ys/Graduate"
DATASET_PATH = os.path.join(ROOT_DIR, "dataset/pheme_source.csv")

# Add model directory to path
sys.path.append(os.path.join(ROOT_DIR, 'model/emotion_roberta'))
from embed_emotions import embed

EMOTION_DIM = 3  # Based on sentiment model output

def load_and_preprocess_data():
    """
    Load and preprocess the PHEME dataset
    """
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    
    # Filter out 'unverified' labels for ADM task
    df = df[df['veracity_label'] != 'unverified']
    
    print(f"Dataset loaded: {len(df)} samples")
    print(f"Veracity labels distribution:")
    print(df['veracity_label'].value_counts())
    
    return df

def preprocess_text(text):
    """
    Preprocess text for tokenization
    """
    if pd.isna(text):
        return ""
    return str(text).strip()

def get_emotion_features(text):
    """
    Get emotion features using the emotion roberta model
    """
    try:
        emotion_tensor = embed(text)
        return emotion_tensor
    except Exception as e:
        print(f"Error getting emotion features for text: {e}")
        # Return zero tensor if emotion extraction fails
        return torch.zeros(EMOTION_DIM)

def encode_labels(df):
    """
    Encode categorical labels to numerical values
    """
    # Encode veracity labels: true -> 1, false -> 0
    veracity_mapping = {'true': 1, 'false': 0}
    df['veracity_encoded'] = df['veracity_label'].map(veracity_mapping)
    
    # Normalize retweet intensity to [0, 1] range
    df['retweet_intensity_normalized'] = df['retweet_intensity'].astype(float)
    
    return df