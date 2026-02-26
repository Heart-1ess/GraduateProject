'''

This script serves the purpose to train a dual model with roberta embedding layer and hpm_adm_dual model.

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
from transformers import XLMRobertaTokenizerFast, BertTokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

ROOT_DIR = "/home/zhangshuhao/projects/ys/Graduate"
DATASET_PATH = os.path.join(ROOT_DIR, "dataset/pheme_source.csv")
LOCAL_ROBERTA_PATH = os.path.join(ROOT_DIR, 'model/hpm_adm_dual/roberta-base')
LOCAL_BERT_PATH = os.path.join(ROOT_DIR, 'model/hpm_adm_dual/bert-base')

sys.path.append(os.path.join(ROOT_DIR, 'model/hpm_adm_dual'))
# Import the dual model
from ablation_models.model_no_emotion import ModelWithOutEmotion
from ablation_models.model_no_roberta import BERTDualChannelModel
from ablation_models.model_single_hpm import HPMModel
from ablation_models.model_single_adm import ADMModel
from hpm_adm_dual.dataset import DualModelDataset
from hpm_adm_dual.utils import *

ABLATION_MODE = ["no_emotion", "no_roberta", "single_hpm", "single_adm"]

# Training parameters
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 20
MAX_LENGTH = 128
EMOTION_DIM = 3  # Based on sentiment model output
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch_dual(model, dataloader, optimizer, criterion_hpm, criterion_adm, device):
    """
    Train the model for one epoch
    """
    model.train()
    total_loss = 0
    hpm_loss_total = 0
    adm_loss_total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        emotion_features = batch['emotion_features'].to(device)
        hpm_labels = batch['hpm_label'].to(device)
        adm_labels = batch['adm_label'].to(device)
        
        # Forward pass
        hpm_pred, adm_pred = model(input_ids, attention_mask, emotion_features)
        
        # Calculate losses
        hpm_loss = criterion_hpm(hpm_pred.squeeze(), hpm_labels)
        adm_loss = criterion_adm(adm_pred.squeeze(), adm_labels)
        
        # Combined loss (weighted)
        total_loss_batch = hpm_loss + adm_loss
        
        # Backward pass
        total_loss_batch.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += total_loss_batch.item()
        hpm_loss_total += hpm_loss.item()
        adm_loss_total += adm_loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Total Loss': f'{total_loss_batch.item():.4f}',
            'HPM Loss': f'{hpm_loss.item():.4f}',
            'ADM Loss': f'{adm_loss.item():.4f}'
        })
    
    return total_loss / len(dataloader), hpm_loss_total / len(dataloader), adm_loss_total / len(dataloader)

def train_epoch_single(model, dataloader, optimizer, criterion, mode, device):
    """
    Train the model for one epoch
    """
    model.train()
    total_loss = 0
    # hpm_loss_total = 0
    # adm_loss_total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        emotion_features = batch['emotion_features'].to(device)
        hpm_labels = batch['hpm_label'].to(device)
        adm_labels = batch['adm_label'].to(device)
        labels = hpm_labels if mode.lower() == "hpm" else adm_labels
        
        # Forward pass
        pred = model(input_ids, attention_mask, emotion_features)
        
        # Calculate losses
        loss = criterion(pred.squeeze(), labels)
        
        # Combined loss (weighted)
        total_loss_batch = loss
        
        # Backward pass
        total_loss_batch.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += total_loss_batch.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Total Loss': f'{total_loss_batch.item():.4f}',
            'Mode': mode
        })
    
    return total_loss / len(dataloader)

def evaluate_model_dual(model, dataloader, criterion_hpm, criterion_adm, device):
    """
    Evaluate the model on validation/test set
    """
    model.eval()
    total_loss = 0
    hpm_loss_total = 0
    adm_loss_total = 0
    
    hpm_predictions = []
    hpm_labels = []
    adm_predictions = []
    adm_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            emotion_features = batch['emotion_features'].to(device)
            hpm_labels_batch = batch['hpm_label'].to(device)
            adm_labels_batch = batch['adm_label'].to(device)
            
            # Forward pass
            hpm_pred, adm_pred = model(input_ids, attention_mask, emotion_features)
            
            # Calculate losses
            hpm_loss = criterion_hpm(hpm_pred.squeeze(), hpm_labels_batch)
            adm_loss = criterion_adm(adm_pred.squeeze(), adm_labels_batch)
            total_loss_batch = hpm_loss + adm_loss
            
            # Update metrics
            total_loss += total_loss_batch.item()
            hpm_loss_total += hpm_loss.item()
            adm_loss_total += adm_loss.item()
            
            # Collect predictions
            hpm_predictions.extend(hpm_pred.squeeze().cpu().numpy())
            hpm_labels.extend(hpm_labels_batch.cpu().numpy())
            adm_predictions.extend(adm_pred.squeeze().cpu().numpy())
            adm_labels.extend(adm_labels_batch.cpu().numpy())
    
    # Calculate metrics
    hpm_mse = np.mean((np.array(hpm_predictions) - np.array(hpm_labels))**2)
    adm_auc = roc_auc_score(adm_labels, adm_predictions)
    
    return {
        'total_loss': total_loss / len(dataloader),
        'hpm_loss': hpm_loss_total / len(dataloader),
        'adm_loss': adm_loss_total / len(dataloader),
        'hpm_mse': hpm_mse,
        'adm_auc': adm_auc
    }

def evaluate_model_single(model, dataloader, criterion, mode, device):
    """
    Evaluate the model on validation/test set
    """
    model.eval()
    total_loss = 0
    # hpm_loss_total = 0
    # adm_loss_total = 0
    
    predictions = []
    labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            emotion_features = batch['emotion_features'].to(device)
            hpm_labels_batch = batch['hpm_label'].to(device)
            adm_labels_batch = batch['adm_label'].to(device)
            labels_batch = hpm_labels_batch if mode.lower() == "hpm" else adm_labels_batch
            
            # Forward pass
            pred = model(input_ids, attention_mask, emotion_features)
            
            # Calculate losses
            loss = criterion(pred.squeeze(), labels_batch)
            total_loss_batch = loss
            
            # Update metrics
            total_loss += total_loss_batch.item()
            
            # Collect predictions
            predictions.extend(pred.squeeze().cpu().numpy())
            labels.extend(labels_batch.cpu().numpy())
    
    # Calculate metrics
    if mode.lower() == 'hpm':
        metrics = np.mean((np.array(predictions) - np.array(labels))**2)
    else:
        metrics = roc_auc_score(labels, predictions)
    
    return {
        'total_loss': total_loss / len(dataloader),
        'mode': mode,
        'score_type': 'mse' if mode.lower() == 'hpm' else 'auc',
        'score': metrics
    }

def main():
    """
    Main training function
    """
    print(f"Using device: {DEVICE}")
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    df = encode_labels(df)
    
    # Split data into train, validation, and test sets
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['veracity_encoded'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['veracity_encoded'])

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Initialize tokenizer
    if ablation_mode == 1:
        tokenizer = BertTokenizer.from_pretrained(LOCAL_BERT_PATH)
        print('bert tokenizer.')
    else:
        tokenizer = XLMRobertaTokenizerFast.from_pretrained(LOCAL_ROBERTA_PATH)
        print('roberta tokenizer.')
    
    # Create datasets
    train_dataset = DualModelDataset(train_df, tokenizer)
    val_dataset = DualModelDataset(val_df, tokenizer)
    test_dataset = DualModelDataset(test_df, tokenizer)
    
    # Create data loaders
    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    match ablation_mode:
        case 0:
            model = ModelWithOutEmotion().to(DEVICE)
            print("train without emotion.")
        case 1:
            model = BERTDualChannelModel().to(DEVICE)
            print("train using bert.")
        case 2:
            model = HPMModel().to(DEVICE)
            print('train single hpm.')
        case 3:
            model = ADMModel().to(DEVICE)
            print('train single adm.')
        case _:
            raise RuntimeError("Invalid mode.")
    
    # Initialize optimizer and loss functions
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Loss functions
    criterion_hpm = nn.MSELoss()  # Regression loss for retweet intensity
    criterion_adm = nn.BCELoss()  # Binary classification loss for veracity
    
    # Training history
    train_losses = []
    val_losses = []
    val_hpm_mse = []
    val_adm_auc = []
    val_metric = []
    val_metric_mode = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 5
    
    print("Starting training...")
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        
        if ablation_mode <= 1:
            print('training dual channel.')
            # Training
            train_loss, train_hpm_loss, train_adm_loss = train_epoch_dual(
                model, train_loader, optimizer, criterion_hpm, criterion_adm, DEVICE
            )
            
            print('evaluate dual channel.')
            # Validation
            val_metrics = evaluate_model_dual(model, val_loader, criterion_hpm, criterion_adm, DEVICE)
        
            # Update scheduler
            scheduler.step(val_metrics['total_loss'])
            
            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_metrics['total_loss'])
            val_hpm_mse.append(val_metrics['hpm_mse'])
            val_adm_auc.append(val_metrics['adm_auc'])
        
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f} (HPM: {train_hpm_loss:.4f}, ADM: {train_adm_loss:.4f})")
            print(f"Val Loss: {val_metrics['total_loss']:.4f}")
            print(f"Val HPM MSE: {val_metrics['hpm_mse']:.4f}")
            print(f"Val ADM AUC: {val_metrics['adm_auc']:.4f}")
        
        else:
            # Training
            if ablation_mode == 2:
                print("train as single hpm.")
                train_loss = train_epoch_single(
                    model, train_loader, optimizer, criterion_hpm, 'hpm', DEVICE
                )
            else:
                print('train as single adm.')
                train_loss = train_epoch_single(
                    model, train_loader, optimizer, criterion_adm, 'adm', DEVICE
                )
            
            # Validation
            if ablation_mode == 2:
                print('validate as single hpm.')
                val_metrics = evaluate_model_single(model, val_loader, criterion_hpm, 'hpm', DEVICE)
            else:
                print('validate as single adm.')
                val_metrics = evaluate_model_single(model, val_loader, criterion_adm, 'adm', DEVICE)
        
            # Update scheduler
            scheduler.step(val_metrics['total_loss'])
            
            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_metrics['total_loss'])
            val_metric.append(val_metrics['score'])
            val_metric_mode.append(val_metrics['score_type'])
        
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['total_loss']:.4f}")
            print(f"Val {'HPM' if ablation_mode == 2 else 'ADM'} {val_metrics['score_type']}: {val_metrics['score']:.4f}")
        
        # Early stopping
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(ROOT_DIR, f'model/hpm_adm_dual/ablation_models/models/best_{ABLATION_MODE[i]}.pth'))
            print("New best model saved!")
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(os.path.join(ROOT_DIR, f'model/hpm_adm_dual/ablation_models/models/best_{ABLATION_MODE[i]}.pth')))
    
    # Final evaluation on test set
    print("\nFinal evaluation on test set:")
    if ablation_mode <= 1:
        test_metrics = evaluate_model_dual(model, test_loader, criterion_hpm, criterion_adm, DEVICE)
        print(f"Test Loss: {test_metrics['total_loss']:.4f}")
        print(f"Test HPM MSE: {test_metrics['hpm_mse']:.4f}")
        print(f"Test ADM AUC: {test_metrics['adm_auc']:.4f}")
    elif ablation_mode == 2:
        test_metrics = evaluate_model_single(model, test_loader, criterion_hpm, 'hpm', DEVICE)
        print(f"Test Loss: {test_metrics['total_loss']:.4f}")
        print(f"Test HPM MSE: {test_metrics['score']:.4f}")
    else:
        test_metrics = evaluate_model_single(model, test_loader, criterion_adm, 'adm', DEVICE)
        print(f"Test Loss: {test_metrics['total_loss']:.4f}")
        print(f"Test ADM AUC: {test_metrics['score']:.4f}")
    
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    if ablation_mode <= 1:
        plt.subplot(1, 3, 2)
        plt.plot(val_hpm_mse)
        plt.title('Validation HPM MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        
        plt.subplot(1, 3, 3)
        plt.plot(val_adm_auc)
        plt.title('Validation ADM AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
    
    else:
        plt.subplot(1, 3, 2)
        plt.plot(val_metric)
        plt.title(f'Validation {"HPM MSE" if ablation_mode == 2 else "ADM AUC"}')
        plt.xlabel('Epoch')
        plt.ylabel("HPM MSE" if ablation_mode == 2 else "ADM AUC")
    
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT_DIR, f'model/hpm_adm_dual/ablation_models/training_curves_{ABLATION_MODE[i]}.png'))
    plt.show()
    
    print("Training completed!")

if __name__ == "__main__":
    for i in range(3, len(ABLATION_MODE)):
        ablation_mode = i
        main()

