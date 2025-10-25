import torch
import torch.utils.data as data
from tqdm import tqdm

import sys
from pathlib import Path
import os

ROOT_DIR = "/home/zhangshuhao/projects/ys/Graduate"

# Add model directory to path
sys.path.append(os.path.join(ROOT_DIR, 'model/hpm_adm_dual/hpm_adm_dual'))
from utils import *

# Training parameters
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 10
MAX_LENGTH = 128

class DualModelDataset(data.Dataset):
    """
    Custom dataset class for dual model training
    """
    def __init__(self, df, tokenizer, max_length=MAX_LENGTH):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print("Preprocessing emotion features...")
        self.emotion_features = []
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            emotion_feat = get_emotion_features(row['text_clean'])
            self.emotion_features.append(emotion_feat)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Tokenize text
        text = preprocess_text(row['text_clean'])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Get emotion features
        emotion_feat = self.emotion_features[idx]
        
        # Get labels
        hpm_label = torch.tensor(row['retweet_intensity_normalized'], dtype=torch.float32)
        adm_label = torch.tensor(row['veracity_encoded'], dtype=torch.float32)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'emotion_features': emotion_feat,
            'hpm_label': hpm_label,
            'adm_label': adm_label,
            'text': text
        }