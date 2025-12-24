import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import warnings
warnings.filterwarnings('ignore')

ROOT_DIR = "/home/zhangshuhao/projects/ys/Graduate"
DATASET_PATH = os.path.join(ROOT_DIR, "dataset/pheme_formatted_with_embeddings.csv")

class ArbitratorDataset(data.Dataset):
    """
    自定义数据集类，用于加载Arbitrator模型的训练数据
    """
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 加载emotion embedding
        emotion_emb_path = row['emotion_emb']
        emotion_emb = np.load(emotion_emb_path)  # shape: (1, 28)
        emotion_emb = emotion_emb.flatten()  # shape: (28,)
        
        # 加载text embedding
        text_emb_path = row['text_emb']
        text_emb = np.load(text_emb_path)  # shape: (32, 768)
        text_emb = text_emb.mean(axis=0)  # 平均池化: (768,)
        
        # 获取propagation_prob和fake_news_prob
        propagation_prob = np.array([row['propagation_prob']], dtype=np.float32)
        fake_news_prob = np.array([row['fake_news_prob']], dtype=np.float32)
        
        # 拼接所有特征
        state = np.concatenate([emotion_emb, text_emb, propagation_prob, fake_news_prob])
        state = torch.tensor(state, dtype=torch.float32)
        
        # 获取标签并确保是整数且在有效范围内 [0, 1]
        label_value = row['veracity_label_id']
        # 转换为整数，确保在[0, 1]范围内
        if pd.isna(label_value):
            label_value = 0  # 默认值
        else:
            label_value = int(float(label_value))
            # 确保标签值在[0, 1]范围内
            if label_value < 0:
                label_value = 0
            elif label_value > 1:
                label_value = 1
        
        label = torch.tensor(label_value, dtype=torch.long)
        
        return {
            'state': state,
            'label': label
        }