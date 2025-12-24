import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import warnings

warnings.filterwarnings("ignore")

ROOT_DIR = "/home/zhangshuhao/projects/ys/Graduate"
DATASET_PATH = os.path.join(ROOT_DIR, "dataset/pheme_formatted_with_embeddings.csv")

# Feature dims
EMOTION_DIM = 28
TEXT_DIM = 768


class EnhancedArbitratorDataset(data.Dataset):
    """
    Dataset for CQL finetuning of the Arbitrator.
    Returns:
        state: Tensor[798]
        action: LongTensor scalar (0=false, 1=true)
        reward: FloatTensor scalar
        is_high_retweet: Bool for evaluation mask
    """

    def __init__(self, df: pd.DataFrame, reward_high_threshold: float = 0.5):
        self.df = df.reset_index(drop=True)
        self.reward_high_threshold = reward_high_threshold

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # load embeddings
        emotion_emb = np.load(row["emotion_emb"]).flatten()  # (28,)
        text_emb = np.load(row["text_emb"]).mean(axis=0)  # (768,)

        propagation_prob = np.array([row["propagation_prob"]], dtype=np.float32)
        fake_news_prob = np.array([row["fake_news_prob"]], dtype=np.float32)

        state_np = np.concatenate([emotion_emb, text_emb, propagation_prob, fake_news_prob])
        state = torch.tensor(state_np, dtype=torch.float32)

        # action/label
        label_value = row["veracity_label_id"]
        if pd.isna(label_value):
            label_value = 0
        label_value = int(float(label_value))
        if label_value < 0:
            label_value = 0
        elif label_value > 1:
            label_value = 1
        action = torch.tensor(label_value, dtype=torch.long)

        # reward shaping: focus on fake & high-retweet samples
        retweet_intensity = float(row.get("retweet_intensity", 0.0))
        is_high_retweet = retweet_intensity > self.reward_high_threshold
        reward = 0.0
        if label_value == 0 and is_high_retweet:
            reward = 1.0
        elif label_value == 0:
            reward = 0.5
        else:
            reward = 0.0

        reward = torch.tensor(reward, dtype=torch.float32)

        return {
            "state": state,
            "action": action,
            "reward": reward,
            "is_high_retweet": torch.tensor(is_high_retweet, dtype=torch.bool),
        }


def load_dataset(csv_path: str = DATASET_PATH, reward_high_threshold: float = 0.5) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # ensure columns
    required = [
        "emotion_emb",
        "text_emb",
        "propagation_prob",
        "fake_news_prob",
        "veracity_label_id",
        "retweet_intensity",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # clean labels
    def clean_label(x):
        if pd.isna(x):
            return 0
        x = int(float(x))
        return max(0, min(1, x))

    df["veracity_label_id"] = df["veracity_label_id"].apply(clean_label)

    # drop rows with missing emb paths
    df = df.dropna(subset=["emotion_emb", "text_emb"])

    return df.reset_index(drop=True)

