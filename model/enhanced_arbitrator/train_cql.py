"""
Conservative Q-Learning (CQL) finetuning for Arbitrator model.
目标：提升假新闻 & 高转发样本的 AUC 与 ACC。
数据来源：dataset/pheme_formatted_with_embeddings.csv
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

ROOT_DIR = "/home/zhangshuhao/projects/ys/Graduate"
sys.path.append(os.path.join(ROOT_DIR, "model/arbitrator"))
sys.path.append(os.path.join(ROOT_DIR, "model/enhanced_arbitrator"))

from arbitrator.model import Arbitrator  # noqa: E402
from enhanced_arbitrator.dataset import (
    EnhancedArbitratorDataset,
    load_dataset,
)  # noqa: E402


# ======================
# Hyperparameters
# ======================
BATCH_SIZE = 64
EPOCHS = 30
LR = 3e-4
ALPHA_CQL = 5.0  # strength of conservative loss
BETA_CE = 1.0  # supervised CE weight
GAMMA = 0.99  # unused for 1-step, kept for extension
REWARD_HIGH_THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# state dimension (emotion 28 + text 768 + 1 + 1)
STATE_DIM = 28 + 768 + 1 + 1
HIDDEN_DIM = 512
ACTION_DIM = 2

MODEL_SAVE_PATH = os.path.join(ROOT_DIR, "model/enhanced_arbitrator/enhanced_arbitrator/best_cql.pth")
PRETRAINED_MODEL_PATH = os.path.join(ROOT_DIR, 'model/arbitrator/arbitrator/base_model251206_loss0.1902_auc0.9738.pth')


def cql_loss(q_values, actions, rewards, alpha=ALPHA_CQL, beta_ce=BETA_CE):
    """
    q_values: [B, A]
    actions: [B] long
    rewards: [B] float
    """
    q_a = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # TD target: one-step with reward only (no next-state info available)
    td_loss = F.mse_loss(q_a, rewards)

    # Conservative term: logsumexp(Q) - Q(a_bc)
    logsumexp_q = torch.logsumexp(q_values, dim=1)
    conservative = (logsumexp_q - q_a).mean()

    # Supervised CE to keep classification ability
    ce = F.cross_entropy(q_values, actions)

    total = td_loss + alpha * conservative + beta_ce * ce
    return total, {"td_loss": td_loss.item(), "cql": conservative.item(), "ce": ce.item()}


def evaluate(model, loader):
    model.eval()
    all_logits = []
    all_labels = []
    all_probs = []
    all_high_mask = []

    with torch.no_grad():
        for batch in loader:
            states = batch["state"].to(DEVICE)
            labels = batch["action"].to(DEVICE)
            high_mask = batch["is_high_retweet"].to(DEVICE)

            logits = model(states)
            probs = F.softmax(logits, dim=-1)[:, 1]

            all_logits.append(logits.cpu())
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
            all_high_mask.append(high_mask.cpu())

    logits = torch.cat(all_logits)
    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    high_mask = torch.cat(all_high_mask)

    preds = torch.argmax(logits, dim=-1)

    metrics = {}
    metrics["acc"] = accuracy_score(labels, preds)
    pr, rc, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    metrics["precision"], metrics["recall"], metrics["f1"] = pr, rc, f1
    metrics["auc"] = roc_auc_score(labels, probs)

    # subset: fake & high retweet
    subset_mask = (labels == 0) & (high_mask == 1)
    if subset_mask.sum() > 0:
        labels_sub = labels[subset_mask]
        probs_sub = probs[subset_mask]
        preds_sub = preds[subset_mask]
        metrics["acc_fake_high"] = accuracy_score(labels_sub, preds_sub)
        pr_s, rc_s, f1_s, _ = precision_recall_fscore_support(
            labels_sub, preds_sub, average="binary", zero_division=0
        )
        metrics["precision_fake_high"], metrics["recall_fake_high"], metrics["f1_fake_high"] = pr_s, rc_s, f1_s
        metrics["auc_fake_high"] = roc_auc_score(labels_sub, probs_sub)
    else:
        metrics["acc_fake_high"] = math.nan
        metrics["precision_fake_high"] = math.nan
        metrics["recall_fake_high"] = math.nan
        metrics["f1_fake_high"] = math.nan
        metrics["auc_fake_high"] = math.nan

    return metrics


def main():
    print(f"Device: {DEVICE}")

    df = load_dataset(reward_high_threshold=REWARD_HIGH_THRESHOLD)
    print(f"Loaded dataset: {len(df)} samples")
    print(df["veracity_label_id"].value_counts())

    dataset = EnhancedArbitratorDataset(df, reward_high_threshold=REWARD_HIGH_THRESHOLD)
    loader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = Arbitrator(state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM).to(DEVICE)
    model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_auc = -1
    best_acc = -1

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        td_collect = []
        cql_collect = []
        ce_collect = []

        for batch in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            states = batch["state"].to(DEVICE)
            actions = batch["action"].to(DEVICE)
            rewards = batch["reward"].to(DEVICE)

            optimizer.zero_grad()
            q_values = model(states)
            loss, parts = cql_loss(q_values, actions, rewards, alpha=ALPHA_CQL, beta_ce=BETA_CE)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            td_collect.append(parts["td_loss"])
            cql_collect.append(parts["cql"])
            ce_collect.append(parts["ce"])

        avg_loss = total_loss / len(loader)
        print(
            f"Epoch {epoch}: loss={avg_loss:.4f} | td={np.mean(td_collect):.4f} | cql={np.mean(cql_collect):.4f} | ce={np.mean(ce_collect):.4f}"
        )

        # Evaluation
        eval_metrics = evaluate(model, loader)
        print(
            f"Eval ACC={eval_metrics['acc']:.4f} AUC={eval_metrics['auc']:.4f} | "
            f"Fake&High ACC={eval_metrics['acc_fake_high']:.4f} AUC={eval_metrics['auc_fake_high']:.4f}"
        )

        # save best by AUC on fake high subset if available else overall
        key_auc = eval_metrics.get("auc_fake_high", np.nan)
        key_acc = eval_metrics.get("acc_fake_high", np.nan)

        better = False
        if not math.isnan(key_auc):
            if key_auc > best_auc or (math.isclose(key_auc, best_auc) and key_acc > best_acc):
                better = True
        else:
            if eval_metrics["auc"] > best_auc or (
                math.isclose(eval_metrics["auc"], best_auc) and eval_metrics["acc"] > best_acc
            ):
                better = True

        if better:
            best_auc = key_auc if not math.isnan(key_auc) else eval_metrics["auc"]
            best_acc = key_acc if not math.isnan(key_acc) else eval_metrics["acc"]
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✓ Saved improved model to {MODEL_SAVE_PATH}")

    print("Training finished.")
    print(f"Best tracked AUC: {best_auc:.4f}, ACC: {best_acc:.4f}")


if __name__ == "__main__":
    main()

