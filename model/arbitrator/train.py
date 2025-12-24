'''
训练脚本：训练Arbitrator模型
输入特征：emotion_emb, text_emb, propagation_prob, fake_news_prob
标签：veracity_label_id (0=假, 1=真)
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

ROOT_DIR = "/home/zhangshuhao/projects/ys/Graduate"
DATASET_PATH = os.path.join(ROOT_DIR, "dataset/pheme_formatted_with_embeddings.csv")

sys.path.append(os.path.join(ROOT_DIR, 'model/arbitrator'))
from arbitrator.model import Arbitrator
from arbitrator.dataset import ArbitratorDataset

# 训练参数
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 50
HIDDEN_DIM = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 特征维度
EMOTION_EMB_DIM = 28  # emotion_emb: (1, 28)
TEXT_EMB_DIM = 768    # text_emb: (32, 768) -> mean pool to 768
PROPAGATION_PROB_DIM = 1
FAKE_NEWS_PROB_DIM = 1
STATE_DIM = EMOTION_EMB_DIM + TEXT_EMB_DIM + PROPAGATION_PROB_DIM + FAKE_NEWS_PROB_DIM  # 798


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    训练一个epoch
    """
    model.train()
    total_loss = 0
    predictions = []
    labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        # 移动数据到设备
        states = batch['state'].to(device)
        labels_batch = batch['label'].to(device)
        
        # 前向传播
        logits = model(states)
        
        # 计算损失
        loss = criterion(logits, labels_batch)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 更新指标
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        predictions.extend(preds.cpu().numpy())
        labels.extend(labels_batch.cpu().numpy())
        
        # 更新进度条
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{accuracy_score(labels, predictions):.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(labels, predictions)
    
    return avg_loss, accuracy


def evaluate_model(model, dataloader, criterion, device):
    """
    评估模型
    """
    model.eval()
    total_loss = 0
    predictions = []
    labels = []
    probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 移动数据到设备
            states = batch['state'].to(device)
            labels_batch = batch['label'].to(device)
            
            # 前向传播
            logits = model(states)
            
            # 计算损失
            loss = criterion(logits, labels_batch)
            
            # 更新指标
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            prob = F.softmax(logits, dim=-1)
            
            predictions.extend(preds.cpu().numpy())
            labels.extend(labels_batch.cpu().numpy())
            probs.extend(prob.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
    
    # 计算AUC
    try:
        auc = roc_auc_score(labels, [p[1] for p in probs])
    except:
        auc = 0.0
    
    # 混淆矩阵
    cm = confusion_matrix(labels, predictions)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm
    }


def main():
    """
    主训练函数
    """
    print(f"使用设备: {DEVICE}")
    print(f"状态维度: {STATE_DIM}")
    
    # 加载数据
    print("加载数据...")
    df = pd.read_csv(DATASET_PATH)
    
    # 检查必要的列是否存在
    required_cols = ['emotion_emb', 'text_emb', 'propagation_prob', 'fake_news_prob', 'veracity_label_id']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要的列: {missing_cols}")
    
    # 移除缺失值
    df = df.dropna(subset=required_cols)
    
    # 清理和验证标签：确保标签是整数且在[0, 1]范围内
    def clean_label(x):
        if pd.isna(x):
            return 0
        x = int(float(x))
        return max(0, min(1, x))  # 限制在[0, 1]范围内
    
    df['veracity_label_id'] = df['veracity_label_id'].apply(clean_label)
    
    print(f"数据样本数: {len(df)}")
    
    # 检查标签分布
    label_counts = df['veracity_label_id'].value_counts().sort_index()
    print(f"标签分布:\n{label_counts}")
    
    # 验证标签值
    unique_labels = sorted(df['veracity_label_id'].unique())
    if not all(label in [0, 1] for label in unique_labels):
        raise ValueError(f"标签值无效: {unique_labels}，应该只包含0和1")
    
    # 划分训练集、验证集和测试集
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.3, 
        random_state=42, 
        stratify=df['veracity_label_id']
    )
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=42, 
        stratify=temp_df['veracity_label_id']
    )
    
    print(f"训练集样本数: {len(train_df)}")
    print(f"验证集样本数: {len(val_df)}")
    print(f"测试集样本数: {len(test_df)}")
    
    # 创建数据集
    train_dataset = ArbitratorDataset(train_df)
    val_dataset = ArbitratorDataset(val_df)
    test_dataset = ArbitratorDataset(test_df)
    
    # 创建数据加载器
    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # 初始化模型
    model = Arbitrator(state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM, action_dim=2).to(DEVICE)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 初始化优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()
    
    # 训练历史
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    val_f1s = []
    val_aucs = []
    
    best_val_f1 = 0.0
    patience_counter = 0
    early_stopping_patience = 10
    
    print("\n开始训练...")
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"{'='*60}")
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        
        # 验证
        val_metrics = evaluate_model(model, val_loader, criterion, DEVICE)
        
        # 更新学习率
        scheduler.step(val_metrics['loss'])
        
        # 存储指标
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_metrics['loss'])
        val_accs.append(val_metrics['accuracy'])
        val_f1s.append(val_metrics['f1'])
        val_aucs.append(val_metrics['auc'])
        
        # 打印epoch结果
        print(f"\n训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"验证 - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        print(f"验证 - Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
        print(f"验证 - F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
        print(f"混淆矩阵:\n{val_metrics['confusion_matrix']}")
        
        # 早停和保存最佳模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            # 保存最佳模型
            model_save_path = os.path.join(ROOT_DIR, 'model/arbitrator/arbitrator/best_model.pth')
            # 确保目录存在
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"✓ 保存最佳模型 (F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            print(f"\n早停触发，在第 {epoch + 1} 个epoch停止训练")
            break
    
    # 加载最佳模型进行最终评估
    print("\n加载最佳模型进行最终评估...")
    model.load_state_dict(torch.load(os.path.join(ROOT_DIR, 'model/arbitrator/arbitrator/best_model.pth')))
    
    # 在测试集上评估
    print("\n在测试集上评估:")
    test_metrics = evaluate_model(model, test_loader, criterion, DEVICE)
    print(f"测试 - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.4f}")
    print(f"测试 - Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}")
    print(f"测试 - F1: {test_metrics['f1']:.4f}, AUC: {test_metrics['auc']:.4f}")
    print(f"混淆矩阵:\n{test_metrics['confusion_matrix']}")
    
    # 绘制训练曲线
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)    
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Training Accuracy', marker='o')
    plt.plot(val_accs, label='Validation Accuracy', marker='s')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(val_f1s, label='Validation F1', marker='o')
    plt.plot(val_aucs, label='Validation AUC', marker='s')
    plt.title('Validation F1 and AUC')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_save_path = os.path.join(ROOT_DIR, 'model/arbitrator/training_curves.png')
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining curves saved to: {plot_save_path}")
    plt.close()
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()

