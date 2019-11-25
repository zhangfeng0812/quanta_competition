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

# global variables
T = 120  # set half year as dynamic period
whole_list = []  # stock cluster prediction
# define the begin time,end time
END_TIME = '2019-10-31'
begin_time = '2018-01-02'
end_time = '2018-12-28'





def fine_tuning_XGBoost():
    X_train, Y_train = get_training_set(begin_time=begin_time, end_time=end_time)
    x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.25, random_state=1)
    xgb1 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
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
    test_accuracy = accuracy_score(y_test, preds)
    print("Test Accuracy: %.2f%%" % (test_accuracy * 100.0))
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
    print(gsearch1.best_params_)
    # 评分
    print(gsearch1.best_score_)
    param_test3 = {
        'gamma': [i / 10.0 for i in range(0, 5)]
    }
    gsearch3 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=4,
                                                    min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                    objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                    seed=27),
                            param_grid=param_test3, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch3.fit(x_train, y_train)
    print(gsearch3.best_params_)
    print(gsearch3.best_score_)
    param_test4 = {
        'subsample': [i / 10.0 for i in range(6, 10)],
        'colsample_bytree': [i / 10.0 for i in range(6, 10)]
    }
    gsearch4 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=177, max_depth=3,
                                                    min_child_weight=4, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
                                                    objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                    seed=27),
                            param_grid=param_test4, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch4.fit(x_train, y_train)
    print(gsearch4.best_params_)
    print(gsearch4.best_score_)


def get_next_day(my_time):
    my_day = datetime.datetime(int(my_time[0:4]), int(my_time[5:7]), int(my_time[8:10]))
    # now = datetime.datetime.now()
    delta = datetime.timedelta(days=+1)
    my_tomorrow = my_day + delta
    my_tomorrow_time = my_tomorrow.strftime('%Y-%m-%d')
    return my_tomorrow_time


def get_week_day(my_time):
    my_day = datetime.datetime(int(my_time[0:4]), int(my_time[5:7]), int(my_time[8:10]))
    # now = datetime.datetime.now()
    delta = datetime.timedelta(days=+7)
    my_tomorrow = my_day + delta
    my_tomorrow_time = my_tomorrow.strftime('%Y-%m-%d')
    return my_tomorrow_time


def get_training_set(begin_time,end_time):
    path = 'data/'
    list_ = []
    for _ in range(9999):
        if begin_time == end_time:
            break
        try:
            df = pd.read_csv(path + begin_time + ".csv", header=0)
        except Exception:
            begin_time = get_day_from_info(begin_time,1)
            continue
        list_.append(df)
        begin_time = get_day_from_info(begin_time,1)
    frame = pd.concat(list_, sort=False)
    shuffle(frame)
    # print(frame)
    # delete some useless columns
    frame = frame.loc[:, ~frame.columns.str.contains('^Unnamed')]
    frame.drop(['date'], axis=1, inplace=True)
    frame.drop(['code'], axis=1, inplace=True)
    frame.drop(['industry_name'], axis=1, inplace=True)
    frame.drop(['closed_price'], axis=1, inplace=True)
    frame.drop(['five_days_closed_price'], axis=1, inplace=True)
    # frame = frame.drop(['index'], axis=1)
    y = frame[['label']]
    frame.drop('label', 1, inplace=True)
    frame.drop('returns', 1, inplace=True)
    return frame, y


def get_test_set(begin_time):
    path = 'data/'
    list_ = []
    for _ in range(9999):
        try:
            df = pd.read_csv(path + str(begin_time) + ".csv", header=0)
            break
        except Exception:
            begin_time = get_day_from_info(begin_time,1)
    list_.append(df)
    frame = pd.concat(list_, sort=False)
    shuffle(frame)
    stock_list = frame['code'].values.tolist()
    frame = frame.loc[:, ~frame.columns.str.contains('^Unnamed')]
    frame.drop(['date'], axis=1, inplace=True)
    frame.drop(['code'], axis=1, inplace=True)
    frame.drop(['industry_name'], axis=1, inplace=True)
    frame.drop(['closed_price'], axis=1, inplace=True)
    frame.drop(['five_days_closed_price'], axis=1, inplace=True)
    # frame = frame.drop(['index'], axis=1)
    frame.drop('label', 1, inplace=True)
    frame.drop('returns', 1, inplace=True)
    return frame, stock_list


def get_day_from_info(related_time,delta):
    db = pymysql.connect(host='192.168.1.252', user='zhangfeng', password='ai436436',
                         port=3306, database='stock_init')
    sql = "select id from trade_days where trade_day like '"+str(related_time)+"'"
    cursor = db.cursor()
    cursor.execute(sql)
    num = cursor.fetchone()
    sql2 = "select trade_day from trade_days where id like '"+str(int(num[0])+delta)+"'"
    cursor.execute(sql2)
    trade_day = cursor.fetchone()
    db.close()
    return trade_day[0]


def predict(clf):
    global end_time,begin_time,END_TIME
    for _ in range(9999):
        if get_day_from_info(end_time,4) == END_TIME:
            break
        if not os.path.exists('data/' + str(end_time) + '.csv'):
            end_time = get_day_from_info(end_time,1)
            continue
        X_train, y_train = get_training_set(begin_time=begin_time, end_time=end_time)
        X_test, stock_list = get_test_set(get_day_from_info(end_time,5))
        clf.fit(X_train, y_train)
        y_pre = clf.predict(X_test)
        y_pro = clf.predict_proba(X_test)[:, 1]
        dic = dict(zip(stock_list, y_pro))
        dic2 = sorted(dic.items(), key=lambda d: d[1], reverse=True)
        temp_list = []
        for k in dic2:
            if k[1] > 0.99:
                temp_list.append(k[0])
            else:
                break
        print(temp_list)
        whole_list.append(temp_list[0:5])
        begin_time = get_day_from_info(begin_time,1)
        end_time = get_day_from_info(end_time,1)

    print(whole_list)








if __name__ == '__main__':
    # test
    # get_day_from_info("2015-01-04",5)
    # dynamic rolling
    # fine_tuning_XGBoost()
    clf = XGBClassifier(
        learning_rate=0.3,  # 默认0.3
        n_estimators=697,  # 树的个数
        max_depth=9,
        min_child_weight=1,
        gamma=0.5,
        subsample=0.6,
        colsample_bytree=0.8,
        objective='binary:logistic',  # 逻辑回归损失函数
        nthread=4,  # cpu线程数
        scale_pos_weight=1,
        seed=27)  # 随机种子
    predict(clf)


