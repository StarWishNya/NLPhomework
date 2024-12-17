import torch
import torch.nn as nn
from torch import optim
from pytorch_models import Model
from pytorch_reasoning import load_model_and_mappings
from config import config
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

cfg = config()

# 读取数据
data_path = "../../data/weibo_words.csv"
data_stop_path = "../../data/stopwords/stopwords.txt"

# 加载数据
data = pd.read_csv(data_path, encoding='utf-8')
labels = data['label'].values
reviews = data['review'].values

# 加载词到索引的映射
model, word_to_index, reverse_label_mapping = load_model_and_mappings()

# 设置词汇表大小
cfg.n_vocab = len(word_to_index)

# 数据预处理
def preprocess_text(text, word_to_index, max_length=32):
    words = text.split()
    indexed_words = [word_to_index.get(word, word_to_index.get('<UNK>', 1)) for word in words]
    if len(indexed_words) > max_length:
        indexed_words = indexed_words[:max_length]
    else:
        indexed_words = indexed_words + [0] * (max_length - len(indexed_words))
    return indexed_words

processed_reviews = [preprocess_text(review, word_to_index) for review in reviews]
reviews_tensor = torch.tensor(processed_reviews, dtype=torch.long)
labels_tensor = torch.tensor(labels, dtype=torch.long)

# 创建数据集和数据加载器
dataset = TensorDataset(reviews_tensor, labels_tensor)
dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

cfg.pad_size = 32

model_text_cls = Model(cfg)
model_text_cls.to(cfg.devices)

# 加载模型, 保存好的模型
model_text_cls.load_state_dict(torch.load("../../model/weibo_pytorch_model.pth"))

# 测试模型
model_text_cls.eval()  # 设置模型为评估模式
total_correct = 0
total_samples = 0

with torch.no_grad():  # 关闭梯度计算
    for i, batch in enumerate(dataloader):
        data, label = batch
        data = data.to(cfg.devices)
        label = label.to(cfg.devices)
        pred_softmax = model_text_cls(data)

        pred = torch.argmax(pred_softmax, dim=1)

        # 统计准确率
        correct = torch.eq(pred, label).sum().item()
        total_correct += correct
        total_samples += label.size(0)

        print(f'Batch {i + 1}: Accuracy = {correct / label.size(0):.4f}')

overall_accuracy = total_correct / total_samples
print(f'Overall Accuracy: {overall_accuracy:.4f}')