import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from src.ML import train_x


def tfmodel():
    # 加载模型
    model = joblib.load('../model/weibo_tfidf_lr.pkl')

    # 加载 TfidfVectorizer
    TF_Vec = TfidfVectorizer(max_df=0.8, min_df=3)
    TF_Vec.fit(train_x)  # 使用训练数据拟合 TfidfVectorizer

    # 输入句子
    input_sentence = input()

    # 转换输入句子
    input_tfvec = TF_Vec.transform([input_sentence])

    # 进行预测
    prediction = model.predict(input_tfvec)

    # 打印预测结果
    print(prediction)


def cvmodel():
    # 加载模型
    model = joblib.load('../model/weibo_count_lr.pkl')

    # 加载 CountVectorizer
    CT_Vec = CountVectorizer(max_df=0.8, min_df=3)
    CT_Vec.fit(train_x)  # 使用训练数据拟合 CountVectorizer

    # 输入句子
    input_sentence = input()

    # 转换输入句子
    input_ctvec = CT_Vec.transform([input_sentence])

    # 进行预测
    prediction = model.predict(input_ctvec)

    # 打印预测结果
    print(prediction)

if __name__ == '__main__':
    choice = input('请选择模型：1.Tfidf 2.CountVectorizer\n')
    while(True):
        if choice == '1':
            tfmodel()
        elif choice == '2':
            cvmodel()
        else:
            break