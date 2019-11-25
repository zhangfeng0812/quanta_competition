# coding:utf-8
import pandas as pd
from sklearn.utils import shuffle
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as mpl
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import datetime
import os
import pymysql
from xgboost import plot_importance
import time
import numpy as np
from AI5_1.data_process import get_last_trade_day
import logging
import method.logging_file




def xgboost_predict():
    for i in range(len(time_list)-T):
        X_train, y_train = get_training_set(i)
        X_test, stock_list = get_test_set(i)
        clf = XGBClassifier(
            learning_rate=0.1,  # 默认0.3
            n_estimators=46,  # 树的个数
            max_depth=5,
            subsample=0.8,
            objective='binary:logistic',  # 逻辑回归损失函数
            nthread=4,  # cpu线程数
            scale_pos_weight=1,
            tree_method='gpu_hist',
            gpu_id=0,
            seed=27)  # 随机种子
        clf.fit(X_train, y_train)
        y_pre = clf.predict(X_test)
        y_pro = clf.predict_proba(X_test)[:, 1]
        dic = dict(zip(stock_list, y_pro))
        dic2 = sorted(dic.items(), key=lambda d: d[1], reverse=True)
        temp_list = []
        for k in dic2:
            temp_list.append(k[0])
        logging.info(temp_list[0:3])
        whole_list.append(temp_list[0:3])


def get_training_set(cnt):
    list_ = []
    for i in range(T):
        df = pd.read_csv(train_data_path + time_list[i+cnt] + ".csv", header=0)
        list_.append(df)
    frame = pd.concat(list_, sort=False)
    frame = shuffle(frame)
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


def get_test_set(cnt):
    logging.info('predict timeline '+str(time_list[T+cnt]))
    list_ = []
    df = pd.read_csv(test_data_path + time_list[T+cnt]+ ".csv", header=0)
    list_.append(df)
    frame = pd.concat(list_, sort=False)
    frame = shuffle(frame)
    stock_list = frame['code'].values.tolist()
    frame = frame.loc[:, ~frame.columns.str.contains('^Unnamed')]
    frame = frame.dropna(axis=0, how='any')
    frame.drop(['date'], axis=1, inplace=True)
    frame.drop(['code'], axis=1, inplace=True)
    frame.drop(['industry_name'], axis=1, inplace=True)
    return frame, stock_list


if __name__ == '__main__':
    method.logging_file.log_file('short_2018_20191')
    whole_list = []
    # get the whole trade date in the last day of the month
    data_path = "E:/factor_data/"
    test_data_path = "E:/factor_data/month_test_data/"
    train_data_path = "E:/factor_data/month_train_data/"
    time_list = get_last_trade_day('2010-01-01', '2019-09-30')
    # define five years as a training period
    T = 60

    try:
        xgboost_predict()
        # get the training set and test set
        # for cnt_inside in range(len(time_list)):
        # fine_tuning_XGBoost(0)
    except Exception:
        pass
    finally:
        logging.info(whole_list)
        print(whole_list)
