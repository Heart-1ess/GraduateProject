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

sys.path.append(os.path.join(ROOT_DIR, 'model/hpm_adm_dual'))
# Import the dual model
from hpm_adm_dual.model import DualChannelModel
from hpm_adm_dual.utils import *

# 示例：模型使用
tokenizer = XLMRobertaTokenizerFast.from_pretrained(LOCAL_ROBERTA_PATH)

# 输入数据（假设我们有文本和情绪特征）
text = "Michael Brown is the 17 yr old boy who was shot 10x &amp; killed by police in #Ferguson today. Media reports \"police shoot man\". #blackboysonly"
emotion_features = torch.unsqueeze(get_emotion_features(text), 0)

inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# 加载模型
model = DualChannelModel()
model.load_state_dict(torch.load(os.path.join(ROOT_DIR, 'model/hpm_adm_dual/hpm_adm_dual/best_model20251025_20epoch_loss0.1194_mse0.0101_auc0.9736.pth')))

model.eval()

# 获取输出
propagation_prob, fake_news_prob = model(input_ids, attention_mask, emotion_features)

print(f"预测人类传播强度: {propagation_prob.item()}")
print(f"信息为真概率: {fake_news_prob.item()}")