import jieba
import pandas as pd
import re
import emoji

def get_words(text):
    if not isinstance(text, str):
        text = ''
    words = jieba.cut(text)
    #清除标点符号
    words = [word for word in words if word.isalnum()]

    #去除停用词
    with open('../../data/stopwords/stopwords.txt', 'r', encoding='utf-8') as f:
        stop_words = f.readlines()
    stop_words = [word.strip() for word in stop_words]
    words = [word for word in words if word not in stop_words]

    return ' '.join(words)  # 将生成器对象转换为字符串

def data_cleaning(text):
    if isinstance(text, str):  # 确保文本为字符串类型
        if '@' in text:  # 如果存在@符号就直接return None
            return None
        text = emoji.replace_emoji(text, replace='')  # 删除表情
        text = re.sub(r'<.*?>', '', text)  # 去除HTML标签
        text = re.sub(r'\s+', ' ', text)  # 去除多余空格
        text = text.strip()
        if text == '' or text.isdigit():  # 排除空消息和只有数字的消息
            return None
        return text
    return text

def cut_short(text):
    if len(text) <= 20:
        return None

if __name__ == '__main__':
    file_path = '../../data/weibo_senti_100k.csv'
    data = pd.read_csv(file_path, encoding='utf-8')

    # 提取评论内容
    review = data['review']
    # 数据清洗
    review = review.apply(data_cleaning)
    # 分词
    data['review'] = review.apply(get_words)
    # 删除评论长度小于20的评论
    data = data.dropna(subset=['review'])
    # 删除空值
    data = data.dropna()
    # 删除空行
    data = data[data['review'] != '']
    # 保存分词结果
    data.to_csv('../data/weibo_words.csv', index=False, columns=['label', 'review'])
    print(data.head())