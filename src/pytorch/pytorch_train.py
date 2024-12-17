import torch
import torch.nn as nn
from torch import optim
import torch.utils.data as Data
import pandas as pd
import numpy as np
import json
import time
from pytorch_models import Model
from config import config


def pad_sequences(sequences, max_length=None, padding='post', truncating='post', pad_value=0):
    """
    统一序列长度的填充函数
    """
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    padded_sequences = np.full((len(sequences), max_length), pad_value, dtype=np.int64)

    for i, seq in enumerate(sequences):
        if len(seq) > max_length:
            if truncating == 'pre':
                seq = seq[-max_length:]
            else:
                seq = seq[:max_length]

        if padding == 'pre':
            padded_sequences[i, -len(seq):] = seq
        else:
            padded_sequences[i, :len(seq)] = seq

    return padded_sequences


def preprocess_text(text, word_to_index, max_length=32):
    """
    将文本转换为整数索引列表
    """
    # 分割文本为词
    words = text.split()
    # 将词转换为对应的索引
    indexed_words = [word_to_index.get(word, word_to_index.get('<UNK>', 1)) for word in words]

    # 截断或填充
    if len(indexed_words) > max_length:
        indexed_words = indexed_words[:max_length]
    else:
        indexed_words = indexed_words + [0] * (max_length - len(indexed_words))

    return indexed_words


def create_word_to_index(reviews):
    """
    从评论中创建词到索引的映射
    """
    # 收集所有唯一词
    all_words = set(' '.join(reviews).split())
    # 创建词到索引的映射
    word_to_index = {
        '<PAD>': 0,  # 填充
        '<UNK>': 1  # 未知词
    }

    for word in all_words:
        if word not in word_to_index:
            word_to_index[word] = len(word_to_index)

    return word_to_index


def train():
    # 读取配置
    cfg = config()

    # 读取数据
    data = pd.read_csv('../../data/weibo_words.csv', encoding='utf-8')

    # 检查并调整标签
    unique_labels = data['label'].unique()
    label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}

    # 使用映射转换标签
    labels = data['label'].map(label_mapping).values

    # 更新配置中的类别数
    cfg.num_classes = len(unique_labels)

    # 创建词到索引的映射
    word_to_index = create_word_to_index(data['review'])

    # 更新词汇表大小
    cfg.n_vocab = len(word_to_index)

    # 数据预处理
    processed_reviews = [preprocess_text(review, word_to_index) for review in data['review']]

    # 转换为张量
    reviews_tensor = torch.tensor(processed_reviews, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # 创建数据集和数据加载器
    dataset = Data.TensorDataset(reviews_tensor, labels_tensor)
    dataloader = Data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.is_shuffle
    )

    # 创建模型
    model = Model(cfg).to(cfg.devices)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=cfg.learn_rate)

    # 训练
    print("开始训练...")
    for epoch in range(cfg.num_epochs):
        model.train()
        total_loss = 0

        for batch_reviews, batch_labels in dataloader:
            # 移动到GPU
            batch_reviews = batch_reviews.to(cfg.devices)
            batch_labels = batch_labels.to(cfg.devices)

            # 清空梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(batch_reviews)

            # 计算损失
            loss = criterion(outputs, batch_labels)

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            total_loss += loss.item()

        # 打印每个epoch的平均损失
        print(f'Epoch [{epoch + 1}/{cfg.num_epochs}], Loss: {total_loss / len(dataloader):.4f}')
        #如果损失小于0.01，停止训练
        if total_loss / len(dataloader) < 0.01:
            break

    # 保存模型
    torch.save(model.state_dict(), '../../model/weibo_pytorch_model.pth')
    label_mapping = {int(label): idx for idx, label in enumerate(sorted(unique_labels))}

    # 保存词到索引的映射和标签映射
    with open('../../model/word_to_index.json', 'w', encoding='utf-8') as f:
        json.dump(word_to_index, f, ensure_ascii=False, indent=2)

    with open('../../model/label_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    start_time = time.time()
    train()
    end_time = time.time()
    print(f'训练时间：{end_time - start_time:.2f}秒')