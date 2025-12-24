'''

This script serves the purpose to use the dual model to prepare embeddings on PHEME dataset for further use.

'''

import os
import sys
import pandas as pd
import numpy as np
import torch
from transformers import XLMRobertaTokenizerFast
import warnings
warnings.filterwarnings('ignore')

ROOT_DIR = "/home/zhangshuhao/projects/ys/Graduate"
DATASET_PATH = os.path.join(ROOT_DIR, "dataset/pheme_formatted.csv")
EMB_PATH = os.path.join(ROOT_DIR, "dataset/PHEME/embeddings")
LOCAL_ROBERTA_PATH = os.path.join(ROOT_DIR, 'model/hpm_adm_dual/roberta-base')

sys.path.append(os.path.join(ROOT_DIR, 'model/hpm_adm_dual'))
# Import the dual model
from hpm_adm_dual.model import DualChannelModel
from hpm_adm_dual.utils import *

# 加载模型
tokenizer = XLMRobertaTokenizerFast.from_pretrained(LOCAL_ROBERTA_PATH)
model = DualChannelModel()
model.load_state_dict(torch.load(os.path.join(ROOT_DIR, 'model/hpm_adm_dual/hpm_adm_dual/best_model20251025_20epoch_loss0.1194_mse0.0101_auc0.9736.pth')))
model.eval()

def emb_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    new_df = pd.DataFrame()
    for idx, row in df.iterrows():
        text = row['text_clean']
        tweet_id = row['tweet_id']
        topic = row['topic']
        # generate emotion embeddings
        emo_emb = torch.unsqueeze(get_emotion_features(text), 0)
        # save the embeddings as npy file
        emo_emb_save_path = os.path.join(EMB_PATH, f"{topic}_{tweet_id}_emotion.npy")
        np.save(emo_emb_save_path, emo_emb)
        row['emotion_emb'] = emo_emb_save_path
        # print(f"Saved emotion embedding for {tweet_id} to {emo_emb_save_path}")

        # get text embeddings
        encoded_input = tokenizer(text, return_tensors='pt')
        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask']
        emotion_features = emo_emb
        text_emb = model.roberta(**encoded_input)[0][0].detach()
        # save the embeddings as npy file
        text_emb_save_path = os.path.join(EMB_PATH, f"{topic}_{tweet_id}_text.npy")
        np.save(text_emb_save_path, text_emb)
        row['text_emb'] = text_emb_save_path
        # print(f"Saved text embedding for {tweet_id} to {text_emb_save_path}")

        # get dual model output
        propagation_prob, fake_news_prob = model(input_ids, attention_mask, emotion_features)
        row['propagation_prob'] = propagation_prob.item()
        row['fake_news_prob'] = fake_news_prob.item()
        # print(f"Saved dual model output for {tweet_id} to {dual_output_save_path}")
        new_df = new_df._append(row, ignore_index=True)
    return new_df

if __name__ == "__main__":
    df = pd.read_csv(DATASET_PATH)
    df = emb_preprocess(df)
    df.to_csv(os.path.join(ROOT_DIR, "dataset/pheme_formatted_with_embeddings.csv"), index=False)