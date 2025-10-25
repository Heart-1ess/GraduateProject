import torch
import torch.nn as nn
from transformers import XLMRobertaModel

import os
ROOT_DIR = "/home/zhangshuhao/projects/ys/Graduate"
LOCAL_ROBERTA_PATH = os.path.join(ROOT_DIR, 'model/hpm_adm_dual/roberta-base')

class DualChannelModel(nn.Module):
    def __init__(self, pretrained_model=LOCAL_ROBERTA_PATH, num_heads=4, embed_dim=768, hidden_dim=256, emotion_dim=28):
        super(DualChannelModel, self).__init__()
        
        # 加载预训练的BERT模型
        self.roberta = XLMRobertaModel.from_pretrained(pretrained_model)
        
        # 人类传播通道（HPM）部分
        # self.lstm_hpm = nn.LSTM(input_size=768, hidden_size=256, bidirectional=True, batch_first=True)
        # self.gate_hpm = nn.Linear(28, 512)
        # self.fc_hpm = nn.Linear(540, 1)  # 输出传播概率

        # Self-Attention 编码器
        self.attention_hpm = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True)
        # Cross Attention 融合（人机双通道）
        self.cross_attn_hpm = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # emotion emb
        self.emotion_emb_hpm = nn.Linear(emotion_dim, embed_dim)
        # 输出层
        self.fc_hpm = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出传播概率（或偏差值）
        )
        
        # AI虚假信息检测通道（ADM）部分
        self.gate_adm = nn.Linear(emotion_dim, embed_dim)
        self.fc_adm = nn.Linear(embed_dim+emotion_dim, 1)  # 输出真假信息判定概率
    
    def forward(self, input_ids, attention_mask, emotion_features):
        # 获取文本的BERT输出
        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = roberta_output.last_hidden_state
        
        # --- 人类传播通道（HPM） ---
        # # LSTM层处理文本上下文
        # lstm_out, _ = self.lstm_hpm(text_embeddings)
        # # 用门控单元获取情感的现行表征并与语义交叉
        # gate_hpm = torch.sigmoid(self.gate_hpm(emotion_features))
        # fusion_hpm = lstm_out[:, -1, :] * gate_hpm
        # combined_hpm = torch.cat((fusion_hpm, emotion_features), dim=1)
        # # 预测传播概率
        # propagation_prob = torch.sigmoid(self.fc_hpm(combined_hpm))  # 输出传播概率（0-1）

        # Self-Attention
        attn_output, attn_weights = self.attention_hpm(text_embeddings, text_embeddings, text_embeddings)

        # emotion emb
        emotion_emb = self.emotion_emb_hpm(emotion_features).unsqueeze(1)
        # Cross-Attention 融合情绪通道
        fused_output, cross_weights = self.cross_attn_hpm(emotion_emb, attn_output, attn_output)
        
        # 平均池化
        fused_output = fused_output.mean(dim=1)
        
        # 输出预测
        propagation_prob = self.fc_hpm(fused_output)
        
        # --- AI虚假信息检测通道（ADM） ---
        # 使用BERT的池化输出并与情绪特征结合
        gate_adm = torch.sigmoid(self.gate_adm(emotion_features))
        fusion_adm = text_embeddings[:, 0, :] * gate_adm
        combined_adm = torch.cat((fusion_adm, emotion_features), dim=1)  # [CLS]位置
        # 预测真假信息
        fake_news_prob = torch.sigmoid(self.fc_adm(combined_adm))  # 输出真假概率（0-1）
        
        return propagation_prob, fake_news_prob
