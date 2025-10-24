'''

This script serves the purpose to use the dual model to inference on tasks.

'''

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from transformers import XLMRobertaTokenizerFast
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

ROOT_DIR = "/home/zhangshuhao/projects/ys/Graduate"
DATASET_PATH = os.path.join(ROOT_DIR, "dataset/pheme_source.csv")
LOCAL_ROBERTA_PATH = os.path.join(ROOT_DIR, 'model/hpm_adm_dual/roberta-base')

# Import the dual model
from hpm_adm_dual.model import DualChannelModel
from hpm_adm_dual.utils import *

# 示例：模型使用
tokenizer = XLMRobertaTokenizerFast.from_pretrained(LOCAL_ROBERTA_PATH)

# 输入数据（假设我们有文本和情绪特征）
text = "This is a sample tweet about the recent event."
emotion_features = get_emotion_features(text)

inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# 加载模型
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DualChannelModel().to(DEVICE)
model.load_state_dict(torch.load(os.path.join(ROOT_DIR, 'model/hpm_adm_dual/hpm_adm_dual/best_model20251024_10epoch_loss0.1330_mse0.0165_auc0.9756.pth')))

# 获取输出
propagation_prob, fake_news_prob = model(input_ids, attention_mask, emotion_features)

print(f"人类传播概率: {propagation_prob.item()}")
print(f"虚假信息判定概率: {fake_news_prob.item()}")