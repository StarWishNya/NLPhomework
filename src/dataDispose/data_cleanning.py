import pandas as pd
import re
import emoji

def clean_text(text):
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

# 读取数据
file_path = '../../data/messages.csv'
data = pd.read_csv(file_path,encoding='utf-8')
# 提取聊天内容
chat_content = data['StrContent']
# 清洗聊天内容
chat_content_cleaned = chat_content.apply(clean_text)
# 删除空值
chat_content_cleaned = chat_content_cleaned.dropna()
# 删除空行
chat_content_cleaned = chat_content_cleaned[chat_content_cleaned != '']
# 保存清洗后的聊天内容
print(chat_content_cleaned.head())
chat_content_cleaned.to_csv('../data/chat_content_cleaned.csv', index=False, header=False) 