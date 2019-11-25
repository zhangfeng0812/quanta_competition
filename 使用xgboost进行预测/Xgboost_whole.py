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

def fine_tuning_XGBoost(cnt_inside):
    para_list = []
    X_train, Y_train = get_training_set(cnt_inside)
    x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.33, random_state=1)
    xgb1 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.9,
        colsample_bytree=0.7,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
    # n_estimators
    eval_set = [(x_test, y_test)]
    xgb1.fit(x_train, y_train, early_stopping_rounds=50, eval_metric="error", eval_set=eval_set,
            verbose=True)  # early_stopping_rounds--当多少次的效果差不多时停止   eval_set--用于显示损失率的数据 verbose--显示错误率的变化过程
    # make prediction
    preds = xgb1.predict(x_test)
    # 确定最好的 n_estimator的值
    n_estimators = xgb1.best_iteration
    para_list.append(n_estimators)
    test_accuracy = accuracy_score(y_test, preds)
    logging.info("Test Accuracy: %.2f%%" % (test_accuracy * 100.0))
    param_test1 = {
        'max_depth': range(3, 10, 2),
        'min_child_weight': range(1, 6, 2)
    }
    gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=n_estimators, max_depth=5,
                                                    min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                    objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                    seed=27),
                            param_grid=param_test1, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch1.fit(x_train, y_train)

    # 最优参数
    depth = gsearch1.best_params_['max_depth']
    child_weight = gsearch1.best_params_['min_child_weight']

    logging.info(gsearch1.best_params_['max_depth'])
    logging.info(gsearch1.best_params_['min_child_weight'])
    para_list.append(depth)
    para_list.append(child_weight)
    # 评分
    logging.info(gsearch1.best_score_)
    param_test3 = {
        'gamma': [i / 10.0 for i in range(0, 5)]
    }
    gsearch3 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=n_estimators, max_depth=depth,
                                                    min_child_weight=child_weight, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                    objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                    seed=27),
                            param_grid=param_test3, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch3.fit(x_train, y_train)
    gamma = gsearch3.best_params_['gamma']
    para_list.append(gamma)
    logging.info(gsearch3.best_params_['gamma'])
    logging.info(gsearch3.best_score_)
    param_test4 = {
        'subsample': [i / 10.0 for i in range(6, 10)],
        'colsample_bytree': [i / 10.0 for i in range(6, 10)]
    }
    gsearch4 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=n_estimators, max_depth=depth,
                                                    min_child_weight=child_weight, gamma=gamma, subsample=0.8, colsample_bytree=0.8,
                                                    objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                    seed=27),
                            param_grid=param_test4, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch4.fit(x_train, y_train)
    logging.info(gsearch4.best_params_['subsample'])
    logging.info(gsearch4.best_params_['colsample_bytree'])
    para_list.append(gsearch4.best_params_['subsample'])
    para_list.append(gsearch4.best_params_['colsample_bytree'])
    logging.info(gsearch4.best_score_)
    return para_list



def xgboost_predict():
    for i in range(len(time_list)-T):
        para_list = fine_tuning_XGBoost(i)
        X_train, y_train = get_training_set(i)
        X_test, stock_list = get_test_set(i)
        clf = XGBClassifier(
            learning_rate=0.1,  # 默认0.3
            n_estimators=para_list[0],  # 树的个数
            max_depth=para_list[1],
            min_child_weight=para_list[2],
            gamma=para_list[3],
            subsample=para_list[4],
            colsample_bytree=para_list[5],
            objective='binary:logistic',  # 逻辑回归损失函数
            nthread=4,  # cpu线程数
            scale_pos_weight=1,
            seed=27)  # 随机种子
        clf.fit(X_train, y_train)
        y_pre = clf.predict(X_test)
        y_pro = clf.predict_proba(X_test)[:, 1]
        dic = dict(zip(stock_list, y_pro))
        dic2 = sorted(dic.items(), key=lambda d: d[1], reverse=True)
        temp_list = []
        for k in dic2:
            temp_list.append(k[0])
        logging.info(temp_list[0:5])
        whole_list.append(temp_list[0:5])


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
    method.logging_file.log_file('xgboost_training_2')
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
