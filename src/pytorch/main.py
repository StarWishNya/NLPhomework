import jieba
import pytorch_reasoning

def cut_text(text):
    # 分词
    words = jieba.cut(text)
    # 去除停用词
    stopwords = [line.strip() for line in open('../../data/stopwords/stopwords.txt', 'r', encoding='utf-8').readlines()]
    filtered_words = [word for word in words if word not in stopwords]
    # 返回处理后的整句话
    return ' '.join(filtered_words)

if __name__ == '__main__':
    # 加载模型和映射
    model, word_to_index, reverse_label_mapping = pytorch_reasoning.load_model_and_mappings()
    while(True):
        # 输入文本
        text = input('请输入文本: ')
        # 分词并处理整句话
        processed_text = cut_text(text)
        # 预测
        prediction, confidence = pytorch_reasoning.predict(processed_text, model, word_to_index, reverse_label_mapping)
        print(f'文本: {processed_text}')
        print(f'情感: {prediction}')
        print(f'置信度: {confidence:.4f}')