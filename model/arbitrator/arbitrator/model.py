import torch
import torch.nn as nn
import torch.nn.functional as F


class Arbitrator(nn.Module):
    """
    双通道人机传播决策模型的策略网络 (Policy Network)
    输入：state（由双通道模型提取的融合特征）
    输出：是否传播的 logits（2 类）
    """

    def __init__(self, state_dim, hidden_dim=512, action_dim=2):
        super(Arbitrator, self).__init__()

        # 第一层：LayerNorm 稳定训练
        self.norm1 = nn.LayerNorm(state_dim)

        # 第一阶段 MLP（提取高维特征）
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # 第二阶段 Residual Block（增强稳定性）
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.res_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.res_fc2 = nn.Linear(hidden_dim, hidden_dim)

        # 输出层（离散动作：传播 vs 不传播）
        self.policy_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """
        x: [batch_size, state_dim]
        """

        # 归一化输入
        x = self.norm1(x)

        # 第一阶段 MLP + GELU
        h = F.gelu(self.fc1(x))
        h = F.gelu(self.fc2(h))

        # 残差块
        r = self.norm2(h)
        r = F.gelu(self.res_fc1(r))
        r = self.res_fc2(r)

        # 残差连接
        h = h + r

        # 输出 logits（未做 softmax，便于 RL）
        logits = self.policy_head(h)

        return logits

    def get_action(self, x):
        """
        推理阶段调用，返回离散动作
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        action = torch.argmax(probs, dim=-1)
        return action, probs
