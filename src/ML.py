import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import time
import joblib

start_time = time.time()

data = pd.read_csv('../data/weibo_words.csv', encoding='utf-8')
data.head()

train_x, test_x, train_y, test_y = model_selection.train_test_split(data['review'], data['label'], test_size=0.2, random_state=0)
print(train_x.shape, test_x.shape)

# 处理 NaN 值
train_x = train_x.fillna('')
test_x = test_x.fillna('')

TF_Vec = TfidfVectorizer(max_df=0.8, min_df=3)
train_x_tfvec = TF_Vec.fit_transform(train_x)
test_x_tfvec = TF_Vec.transform(test_x)

CT_Vec = CountVectorizer(max_df=0.8, min_df=3)
train_x_ctvec = CT_Vec.fit_transform(train_x)
test_x_ctvec = CT_Vec.transform(test_x)

