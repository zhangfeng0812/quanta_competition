# coding:utf-8
import pymysql
import pandas as pd
import numpy as np
import csv
from sklearn.utils import shuffle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, roc_auc_score
import scipy.sparse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as mpl
from xgboost import XGBClassifier
from sklearn.datasets import load_svmlight_file  # 用于直接读取svmlight文件形式， 否则就需要使用xgboost.DMatrix(文件名)来读取这种格式的文件
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from method.logging_file import log_file
from AI5_1.data_process import get_last_trade_day

def get_training_set(cnt):
    list_ = []
    for i in range(T):
        try:
            df = pd.read_csv(train_data_path + time_list[i+cnt] + ".csv", header=0)
            list_.append(df)
        except Exception:
            pass
    frame = pd.concat(list_, sort=False)
    frame = shuffle(frame,random_state=1)
    # logging.info(frame)
    # delete some useless columns
    frame = frame.loc[:, ~frame.columns.str.contains('^Unnamed')]
    frame = frame.dropna(axis=0, how='any')
    frame.drop(['date'], axis=1, inplace=True)
    frame.drop(['code'], axis=1, inplace=True)
    frame.drop(['industry_name'], axis=1, inplace=True)
    frame.drop(['closed_price'], axis=1, inplace=True)
    frame.drop(['next_months_closed_price'], axis=1, inplace=True)
    # frame = frame.drop(['index'], axis=1)
    y = frame[['label']]
    frame.drop('label', 1, inplace=True)
    frame.drop('returns', 1, inplace=True)
    return frame, y



def main3():
    x_train, x_test, y_train, y_test = train_test_split(frame, Y, test_size=0.25, random_state=1)
    num_round = 46
    eval_set = [(x_test, y_test)]
    bst2 = XGBClassifier(max_depth=8, subsample=0.8, learning_rate=0.1, n_estimators=num_round, silent=True,
                         objective='binary:logistic',tree_method='gpu_hist',gpu_id=0)
    bst2.fit(x_train, y_train, early_stopping_rounds=20, eval_metric="auc", eval_set=eval_set,
             verbose=True)
    feature = bst2.feature_importances_.tolist()
    column_name = x_train.columns.values.tolist()
    # from xgboost import plot_importance  # 显示特征重要性
    # plot_importance(bst2)  # 打印重要程度结果。
    # pyplot.show()
    df.insert(df.shape[1], time_list[temp*12][0:4], feature)
    try:
        df.index = column_name
    except Exception:
        pass


if __name__ == '__main__':
    log_file('xgboost_importance')
    T = 12
    data_path = "E:/factor_data/"
    test_data_path = "E:/factor_data/month_test_data/"
    train_data_path = "E:/factor_data/month_train_data/"
    time_list = get_last_trade_day('2015-01-01', '2019-10-31')
    df = pd.DataFrame()
    for temp in range(0,5):
        frame,Y = get_training_set(temp*12)
        main3()
    print(df)
    df.insert(df.shape[1], "均值", df.mean(1))
    print(1)
    df = df.sort_values(by="均值", ascending=False)
    df.to_excel("1.xls")
