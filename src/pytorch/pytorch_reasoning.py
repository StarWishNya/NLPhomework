import torch
import json
from pytorch_models import Model
from config import config


def load_model_and_mappings(model_path='../../model/weibo_pytorch_model.pth',
                            word_to_index_path='../../model/word_to_index.json',
                            label_mapping_path='../../model/label_mapping.json'):
    """
    加载模型、词到索引的映射和标签映射
    """
    # 加载词到索引的映射
    with open(word_to_index_path, 'r', encoding='utf-8') as f:
        word_to_index = json.load(f)

    # 加载标签映射
    with open(label_mapping_path, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)

    # 反转标签映射（用于将预测的索引转换回原始标签）
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}

    # 初始化配置
    cfg = config()
    cfg.n_vocab = len(word_to_index)
    cfg.num_classes = len(label_mapping)

    # 创建模型
    model = Model(cfg)

    # 加载模型权重
    model.load_state_dict(torch.load(model_path))

    # 设置为评估模式
    model.eval()

    return model, word_to_index, reverse_label_mapping


def preprocess_text(text, word_to_index, max_length=32):
    """
    将文本转换为模型可接受的输入格式
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

    return torch.tensor(indexed_words, dtype=torch.long).unsqueeze(0)  # 添加batch维度


def predict(text, model, word_to_index, reverse_label_mapping):
    """
    对单个文本进行预测
    """
    # 预处理文本
    input_tensor = preprocess_text(text, word_to_index)

    # 关闭梯度计算
    with torch.no_grad():
        # 模型预测
        outputs = model(input_tensor)

        # 获取预测的类别
        _, predicted = torch.max(outputs, 1)

        # 获取预测的概率
        probabilities = torch.softmax(outputs, dim=1)

    # 转换预测结果
    predicted_label = reverse_label_mapping[predicted.item()]
    confidence = probabilities[0][predicted.item()].item()

    return predicted_label, confidence


def main():
    # 加载模型和映射
    model, word_to_index, reverse_label_mapping = load_model_and_mappings()

    # 测试文本
    test_texts = [
        "更博 爱 生活",
        "问题 困难",
        "开心 快乐 笑"
    ]

    # 进行预测
    for text in test_texts:
        predicted_label, confidence = predict(text, model, word_to_index, reverse_label_mapping)
        print(f"文本: {text}")
        print(f"预测标签: {predicted_label}")
        print(f"置信度: {confidence:.4f}\n")