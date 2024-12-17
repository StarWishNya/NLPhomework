import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embeding = nn.Embedding(config.n_vocab,
                                     embedding_dim=config.embed_size,
                                     padding_idx=0)  # 使用0作为填充索引

        self.lstm = nn.LSTM(input_size=config.embed_size,
                            hidden_size=config.hidden_size,
                            num_layers=config.num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=config.dropout)

        # 计算全连接层输入大小
        fc_input_size = config.hidden_size * 2 + config.embed_size

        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, config.num_classes)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 嵌入
        embed = self.embeding(x)  # (batch_size, seq_len, embed_size)

        # LSTM处理
        lstm_out, (hidden, _) = self.lstm(embed)

        # 取最后一个时间步的隐藏状态
        hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)  # 连接两个方向的最后隐藏状态

        # 全局平均池化
        embed_mean = embed.mean(dim=1)

        # 拼接特征
        combined = torch.cat([hidden, embed_mean], dim=1)

        # 全连接层
        out = self.fc(combined)

        return out