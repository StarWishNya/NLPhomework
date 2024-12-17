from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import time
import joblib
import numpy as np
from src.ML import train_y, test_y, train_x_ctvec,test_x_ctvec

if __name__ == '__main__':
    start_time = time.time()
    lr = linear_model.LogisticRegression(penalty='l2', C=1.0, solver='liblinear', max_iter=1000, multi_class='ovr')
    model = GridSearchCV(lr, cv=3, param_grid={'C': np.logspace(0, 4, 30), 'penalty': ['l1', 'l2']})
    model.fit(train_x_ctvec, train_y)
    print('最优参数：', model.best_params_)
    pre_train_y = model.predict(train_x_ctvec)
    train_accuracy = accuracy_score(train_y, pre_train_y)
    pre_test_y = model.predict(test_x_ctvec)
    test_accuracy = accuracy_score(test_y, pre_test_y)
    print('训练集准确率：', train_accuracy)
    print('测试集准确率：', test_accuracy)
    end_time = time.time()
    print('耗时：', end_time - start_time)
    joblib.dump(model, '../model/weibo_count_lr.pkl')