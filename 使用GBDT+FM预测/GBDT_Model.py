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
import datetime
import os
import pymysql
from xgboost import plot_importance
import time
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from AI5_1.data_process import get_last_trade_day
import logging
import method.logging_file
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

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

def gbdt():
    def sub_sample_fine_tuning(estimator, max_depth, min_samples_split, min_samples_leaf, max_features, frame_train,
                               Y):
        param_test5 = {'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]}
        grid_search5 = GridSearchCV(
            estimator=GradientBoostingClassifier(learning_rate=0.1,
                                                 n_estimators=estimator, max_depth=max_depth,
                                                 min_samples_leaf=min_samples_leaf,
                                                 min_samples_split=min_samples_split, max_features=max_features,
                                                 random_state=10),
            param_grid=param_test5,
            scoring='roc_auc',
            iid=False,
            cv=5
        )
        grid_result5 = grid_search5.fit(frame_train, Y)
        ##打印结果
        logging.info("Best: %f using %s" % (grid_result5.best_score_, grid_result5.best_params_))
        means = grid_result5.cv_results_['mean_test_score']
        params = grid_result5.cv_results_['params']
        for mean, param in zip(means, params):
            logging.info("mean:  %f  , params:  %r" % (mean, param))
        return grid_result5.best_params_['subsample']

    def max_features_fine_tuning(estimator, max_depth, min_samples_split, min_samples_leaf,frame_train, Y):
        param_test4 = {'max_features': range(7, 20, 2)}
        grid_search4 = GridSearchCV(
            estimator=GradientBoostingClassifier(learning_rate=0.1,
                                                 n_estimators=estimator, max_depth=max_depth,
                                                 min_samples_leaf=min_samples_leaf,
                                                 min_samples_split=min_samples_split, subsample=0.8, random_state=10),
            param_grid=param_test4,
            scoring='roc_auc',
            iid=False,
            cv=5
        )
        grid_result4 = grid_search4.fit(frame_train, Y)
        ##打印结果
        logging.info("Best: %f using %s" % (grid_result4.best_score_, grid_result4.best_params_))
        means = grid_result4.cv_results_['mean_test_score']
        params = grid_result4.cv_results_['params']
        for mean, param in zip(means, params):
            logging.info("mean:  %f  , params:  %r" % (mean, param))
        return grid_result4.best_params_['max_features']

    def new_para_fine_tuning(estimator, max_depth, min_samples_split, min_samples_leaf,frame_train, Y):
        gbm1 = GradientBoostingClassifier(
            learning_rate=0.1, n_estimators=estimator, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split, max_features='sqrt', subsample=0.8, random_state=10
        )
        gbm1.fit(frame_train, Y)
        y_pred = gbm1.predict(frame_train)
        y_predprob = gbm1.predict_proba(frame_train)[:, 1]  # 样本预测为类别1的概率（默认使用正样本标签计算AUC）
        logging.info("Accuracy:%.4f" % metrics.accuracy_score(Y, y_pred))
        logging.info("AUC Score(Train):%f" % metrics.roc_auc_score(Y, y_predprob))

    def min_sample_split_leaf_fine_tuning(estimator, max_depth,frame_train, Y):
        param_test3 = {'min_samples_split': range(800, 1900, 200), 'min_samples_leaf': range(40, 81, 10)}
        grid_search3 = GridSearchCV(
            estimator=GradientBoostingClassifier(learning_rate=0.1,
                                                 n_estimators=estimator, max_depth=max_depth,
                                                 max_features='sqrt', subsample=0.8, random_state=10),
            param_grid=param_test3,
            scoring='roc_auc',
            iid=False,
            cv=5
        )
        grid_result3 = grid_search3.fit(frame_train, Y)
        ##打印结果
        logging.info("Best: %f using %s" % (grid_result3.best_score_, grid_result3.best_params_))
        means = grid_result3.cv_results_['mean_test_score']
        params = grid_result3.cv_results_['params']
        for mean, param in zip(means, params):
            logging.info("mean:  %f  , params:  %r" % (mean, param))
        return grid_result3.best_params_['min_samples_split'], grid_result3.best_params_['min_samples_leaf']

    def max_depth_min_sample_split_fine_tuning(estimator,frame_train, Y):
        param_test2 = {'max_depth': range(3, 14, 2), 'min_samples_split': range(100, 801, 200)}
        grid_search2 = GridSearchCV(
            estimator=GradientBoostingClassifier(learning_rate=0.1,
                                                 n_estimators=estimator, min_samples_leaf=20,
                                                 max_features='sqrt', subsample=0.8, random_state=10),
            param_grid=param_test2,
            scoring='roc_auc',
            iid=False,
            cv=5
        )
        grid_result2 = grid_search2.fit(frame_train, Y)
        ##打印结果
        logging.info("Best: %f using %s" % (grid_result2.best_score_, grid_result2.best_params_))
        means = grid_result2.cv_results_['mean_test_score']
        params = grid_result2.cv_results_['params']
        for mean, param in zip(means, params):
            logging.info("mean:  %f  , params:  %r" % (mean, param))
        return grid_result2.best_params_['max_depth'], grid_result2.best_params_['min_samples_split']

    def n_estimator_fine_tuning(frame_train, Y):
        param_test1 = {'n_estimators': range(50, 281, 30)}
        grid_search1 = GridSearchCV(
            estimator=GradientBoostingClassifier(learning_rate=0.1,
                                                 min_samples_split=300, min_samples_leaf=20, max_depth=8,
                                                 max_features='sqrt', subsample=0.8, random_state=10),
            param_grid=param_test1,
            scoring='roc_auc',
            iid=False,
            cv=5
        )
        grid_result1 = grid_search1.fit(frame_train, Y)
        ##打印结果
        logging.info( "Best: %f using %s" % (grid_result1.best_score_, grid_result1.best_params_))
        means = grid_result1.cv_results_['mean_test_score']
        params = grid_result1.cv_results_['params']

        for mean, param in zip(means, params):
            logging.info("mean:  %f  , params:  %r" % (mean, param))  ##%r是万能格式符，会将后面参数原样打印出来
        return grid_result1.best_params_['n_estimators']

    def fine_tuning_without_para(frame_train, Y):
        gbm0 = GradientBoostingClassifier(random_state=10)
        gbm0.fit(frame_train, Y)
        y_pred = gbm0.predict(frame_train)
        y_predprob = gbm0.predict_proba(frame_train)[:, 1]  # 样本预测为类别1的概率（默认使用正样本标签计算AUC）
        logging.info(":Accuracy:%.4f" % metrics.accuracy_score(Y, y_pred))
        logging.info("AUC Score(Train):%f" % metrics.roc_auc_score(Y, y_predprob))

    def main():
        fine_tuning_without_para(X_train, Y_train)
        estimator = n_estimator_fine_tuning(X_train, Y_train)
        max_depth, min_samples_split = max_depth_min_sample_split_fine_tuning(estimator, X_train, Y_train)
        min_samples_split, min_samples_leaf = min_sample_split_leaf_fine_tuning(estimator, max_depth,X_train, Y_train)
        new_para_fine_tuning(estimator, max_depth, min_samples_split, min_samples_leaf, X_train, Y_train)
        max_features = max_features_fine_tuning(estimator, max_depth, min_samples_split, min_samples_leaf, X_train, Y_train)
        sub_sample = sub_sample_fine_tuning(estimator, max_depth, min_samples_split, min_samples_leaf, max_features, X_train, Y_train)
        grd = GradientBoostingClassifier(n_estimators=100, random_state=10, subsample=0.6, max_depth=7,
                                         min_samples_split=900)
        # 调用one-hot编码。
        grd_enc = OneHotEncoder()
        # 调用LR分类模型。
        grd_lm = LogisticRegression()
        '''使用X_train训练GBDT模型，后面用此模型构造特征'''
        x_train, x_train_lr, y_train, y_train_lr = train_test_split(X_train, Y_train, test_size=0.25)
        grd.fit(x_train, y_train)
        # fit one-hot编码器
        grd_enc.fit(grd.apply(x_train)[:, :, 0])
        '''
        使用训练好的GBDT模型构建特征，然后将特征经过one-hot编码作为新的特征输入到LR模型训练。
        '''
        temp = grd_enc.transform(grd.apply(x_train_lr)[:, :, 0])
        temp2 = temp.todense()
        temp3 = np.asarray(temp2)
        # print(temp3)
        # write a form of ffm model
        temp = grd_enc.transform(grd.apply(x_train_lr)[:, :, 0])
        temp2 = temp.todense()
        temp3 = np.asarray(temp2)
        # print(temp3)
        # write a form of fm model
        path = "E:/factor_data/small_train/" + str(time_now) + ".txt"
        if not os.path.exists(path):
            with open(path, "w", encoding='utf-8') as f1:
                print('the file is already set up')
        with open(path, "w", encoding='utf-8') as f2:
            for i in range(temp3.shape[0]):
                f2.write(str(int(y_train_lr.iloc[i][0])) + '\t')
                for j in range(temp3.shape[1]):
                    f2.write(str(int(temp3[i][j])) + ' ')
                f2.write('\n')
        path2 = "E:/factor_data/small_test/" + str(time_now) + ".txt"
        temp = grd_enc.transform(grd.apply(X_test)[:, :, 0])
        temp2 = temp.todense()
        temp3 = np.asarray(temp2)
        if not os.path.exists(path2):
            with open(path2, "w", encoding='utf-8') as f1:
                print('the file is already set up')
        with open(path2, "w", encoding='utf-8') as f2:
            for i in range(temp3.shape[0]):
                for j in range(temp3.shape[1]):
                    f2.write(str(int(temp3[i][j])) + ' ')
                f2.write('\n')
        grd_lm.fit(grd_enc.transform(grd.apply(x_train_lr)[:, :, 0]), y_train_lr)
        # 用训练好的LR模型多X_test做预测
        y_pred_grd_lm = grd_lm.predict_proba(grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1]
        # 根据预测结果输出
        y_predict = y_pred_grd_lm.tolist()
        dic = dict(zip(stock_list, y_predict))
        dic2 = sorted(dic.items(), key=lambda d: d[1], reverse=True)
        temp_list = []
        for k in dic2:
            temp_list.append(k[0])
        gbdt_list.append(temp_list[0:5])
    main()

if __name__ == '__main__':
    method.logging_file.log_file('gbdt_1')
    whole_list = []
    gbdt_list = []
    # get the whole trade date in the last day of the month
    data_path = "E:/factor_data/"
    test_data_path = "E:/factor_data/month_test_data/"
    train_data_path = "E:/factor_data/month_train_data/"
    time_list = get_last_trade_day('2010-01-01', '2019-09-30')
    # define five years as a training period
    T = 12
    time_now = -1
    for i in range(len(time_list)-T):
        time_now = time_list[i]
        try:
            X_train, Y_train = get_training_set(i)
            X_test,stock_list = get_test_set(i)
            gbdt()
            # get the training set and test set
            # for cnt_inside in range(len(time_list)):
            # fine_tuning_XGBoost(0)
        except Exception:
            pass
        finally:
            logging.info(whole_list)
            print(whole_list)
            print(gbdt_list)
