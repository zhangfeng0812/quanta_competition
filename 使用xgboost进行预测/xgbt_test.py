# coding:utf-8
import pymysql
import pandas as pd
import numpy as np
import csv
from sklearn.utils import shuffle
import xgboost as xgb
from sklearn.model_selection import train_test_split
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


def main():
    path = 'data/'
    frame = pd.DataFrame()
    list_ = []
    for i in range(1, 84):
        df = pd.read_csv(path + str(i) + ".csv", index_col=0, header=0)
        list_.append(df)
    frame = pd.concat(list_)
    # print(frame)
    # delete some useless columns
    frame = frame.drop(['trade_date'], axis=1)
    frame = frame.drop(['stock_code'], axis=1)
    frame = frame.drop(['industry_type'], axis=1)
    # frame = frame.drop(['index'], axis=1)
    frame = frame.drop(['closed_price'], axis=1)
    frame = shuffle(frame)
    Y = frame[['label']]
    Y.loc[Y['label'] == -1] = 0
    X = frame.drop('label', 1,inplace =True)
    X = frame.drop('returns', 1,inplace = True)
    # print(frame)
    # the following steps are constructed X_TRAIN, Y_TRAIN, X_TEST, Y_TEST
    # 　print(X_)
    x_train,x_test,y_train,y_test=train_test_split(frame,Y,test_size=0.25,random_state=1)  #训练集和测试集
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(y_test, label=y_test)
    param = {'max_depth': 2, 'eta': 0.6, 'silent': 0, 'objective': 'binary:logistic'}
    num_round = 2
    bst = xgb.train(param, dtrain, num_round)
    train_preds = bst.predict(dtrain)
    train_predictions = [round(value) for value in train_preds]  # 进行四舍五入的操作--变成0.1(算是设定阈值的符号函数)
    train_accuracy = accuracy_score(y_train, train_predictions)  # 使用sklearn进行比较正确率
    print("Train Accuary: %.2f%%" % (train_accuracy * 100.0))

    from xgboost import plot_importance  # 显示特征重要性
    plot_importance(bst)  # 打印重要程度结果。
    mpl.show()
def main2():
    # 未设定早停止， 未进行矩阵变换
    path = 'data/'
    frame = pd.DataFrame()
    list_ = []
    for i in range(1, 84):
        df = pd.read_csv(path + str(i) + ".csv", index_col=0, header=0)
        list_.append(df)
    frame = pd.concat(list_)
    # print(frame)
    # delete some useless columns
    frame = frame.drop(['trade_date'], axis=1)
    frame = frame.drop(['stock_code'], axis=1)
    frame = frame.drop(['industry_type'], axis=1)
    # frame = frame.drop(['index'], axis=1)
    frame = frame.drop(['closed_price'], axis=1)
    frame = shuffle(frame)
    Y = frame[['label']]
    Y.loc[Y['label'] == -1] = 0
    X = frame.drop('label', 1,inplace = True)
    X = frame.drop('returns', 1,inplace =True)
    # print(frame)
    # the following steps are constructed X_TRAIN, Y_TRAIN, X_TEST, Y_TEST
    # 　print(X_)
    x_train, x_test, y_train, y_test = train_test_split(frame, Y, test_size=0.25, random_state=1)
    num_round = 100
    bst1 = XGBClassifier(max_depth=2, learning_rate=1, n_estimators=num_round,  # 弱分类树太少的话取不到更多的特征重要性
                         silent=True, objective='binary:logistic')
    bst1.fit(x_train, y_train)

    train_preds = bst1.predict(x_train)
    train_accuracy = accuracy_score(y_train, train_preds)
    print("Train Accuary: %.2f%%" % (train_accuracy * 100.0))

    preds = bst1.predict(x_test)
    test_accuracy = accuracy_score(y_test, preds)
    print("Test Accuracy: %.2f%%" % (test_accuracy * 100.0))

    from xgboost import plot_importance  # 显示特征重要性
    plot_importance(bst1)  # 打印重要程度结果。
    pyplot.show()


def main3():
    path = 'data/'
    frame = pd.DataFrame()
    list_ = []
    for i in range(1, 84):
        df = pd.read_csv(path + str(i) + ".csv", index_col=0, header=0)
        list_.append(df)
    frame = pd.concat(list_)
    # print(frame)
    # delete some useless columns
    frame = frame.drop(['trade_date'], axis=1)
    frame = frame.drop(['stock_code'], axis=1)
    frame = frame.drop(['industry_type'], axis=1)
    # frame = frame.drop(['index'], axis=1)
    frame = frame.drop(['closed_price'], axis=1)
    frame = shuffle(frame)
    Y = frame[['label']]
    Y.loc[Y['label'] == -1] = 0
    X = frame.drop('label', 1,inplace=True)
    X = frame.drop('returns', 1,inplace=True)
    # print(frame)
    # the following steps are constructed X_TRAIN, Y_TRAIN, X_TEST, Y_TEST
    # 　print(X_)
    x_train, x_test, y_train, y_test = train_test_split(frame, Y, test_size=0.25, random_state=1)
    param = {'max_depth': 2, 'eta': 1, 'silent': 0, 'objective': 'binary:logistic'}
    num_round = 100
    bst2 = XGBClassifier(max_depth=2, learning_rate=0.1, n_estimators=num_round, silent=True,
                         objective='binary:logistic')
    bst2.fit(x_train, y_train)
    kfold = StratifiedKFold(n_splits=10, random_state=7)
    results = cross_val_score(bst2, x_train, y_train, cv=kfold)  # 对数据进行十折交叉验证--9份训练，一份测试

    print(results)
    print("CV Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

    from xgboost import plot_importance  # 显示特征重要性
    plot_importance(bst2)  # 打印重要程度结果。
    pyplot.show()
def main4():
    # 使用sklearn中提供的网格搜索进行测试--找出最好参数，并作为默认训练参数
    path = 'data/'
    frame = pd.DataFrame()
    list_ = []
    for i in range(1, 84):
        df = pd.read_csv(path + str(i) + ".csv", index_col=0, header=0)
        list_.append(df)
    frame = pd.concat(list_)
    # print(frame)
    # delete some useless columns
    frame = frame.drop(['trade_date'], axis=1)
    frame = frame.drop(['stock_code'], axis=1)
    frame = frame.drop(['industry_type'], axis=1)
    # frame = frame.drop(['index'], axis=1)
    frame = frame.drop(['closed_price'], axis=1)
    frame = shuffle(frame)
    Y = frame[['label']]
    Y.loc[Y['label'] == -1] = 0
    X = frame.drop('label', 1, inplace=True)
    X = frame.drop('returns', 1, inplace=True)
    # print(frame)
    # the following steps are constructed X_TRAIN, Y_TRAIN, X_TEST, Y_TEST
    # 　print(X_)
    x_train, x_test, y_train, y_test = train_test_split(frame, Y, test_size=0.25, random_state=1)
    params = {'max_depth': 2, 'eta': 0.1, 'silent': 0, 'objective': 'binary:logistic'}
    bst = XGBClassifier(max_depth=2, learning_rate=0.1, silent=True, objective='binary:logistic')
    param_test = {
        'n_estimators': range(1, 51, 1)
    }
    clf = GridSearchCV(estimator=bst, param_grid=param_test, scoring='accuracy', cv=5)  # 5折交叉验证
    clf.fit(x_train, y_train)  # 默认使用最优的参数

    preds = clf.predict(x_test)

    test_accuracy = accuracy_score(y_test, preds)
    print("Test Accuracy of gridsearchcv: %.2f%%" % (test_accuracy * 100.0))

    print(clf.cv_results_)
    print(clf.best_params_)
    print(clf.best_score_)


def main5():
    path = 'data/'
    frame = pd.DataFrame()
    list_ = []
    for i in range(1, 84):
        df = pd.read_csv(path + str(i) + ".csv", index_col=0, header=0)
        list_.append(df)
    frame = pd.concat(list_)
    # print(frame)
    # delete some useless columns
    frame = frame.drop(['trade_date'], axis=1)
    frame = frame.drop(['stock_code'], axis=1)
    frame = frame.drop(['industry_type'], axis=1)
    # frame = frame.drop(['index'], axis=1)
    frame = frame.drop(['closed_price'], axis=1)
    frame = shuffle(frame)
    Y = frame[['label']]
    Y.loc[Y['label'] == -1] = 0
    X = frame.drop('label', 1, inplace=True)
    X = frame.drop('returns', 1, inplace=True)
    # print(frame)
    # the following steps are constructed X_TRAIN, Y_TRAIN, X_TEST, Y_TEST
    # 　print(X_)
    x_train, x_test, y_train, y_test = train_test_split(frame, Y, test_size=0.25, random_state=1)
    # 进行提早停止的单独实例

    param = {'max_depth': 2, 'eta': 1, 'silent': 0, 'objective': 'binary:logistic'}
    num_round = 100
    # num_round = 44
    bst = XGBClassifier(max_depth=2, learning_rate=0.1, n_estimators=num_round, silent=True,
                        objective='binary:logistic')
    eval_set = [(x_test, y_test)]
    bst.fit(x_train, y_train, early_stopping_rounds=10, eval_metric="error", eval_set=eval_set,
            verbose=True)  # early_stopping_rounds--当多少次的效果差不多时停止   eval_set--用于显示损失率的数据 verbose--显示错误率的变化过程
    # make prediction
    preds = bst.predict(x_test)

    test_accuracy = accuracy_score(y_test, preds)
    print("Test Accuracy: %.2f%%" % (test_accuracy * 100.0))

def main6():
    # 多参数顺
    path = 'data/'
    frame = pd.DataFrame()
    list_ = []
    for i in range(1, 84):
        df = pd.read_csv(path + str(i) + ".csv", index_col=0, header=0)
        list_.append(df)
    frame = pd.concat(list_)
    # print(frame)
    # delete some useless columns
    frame = frame.drop(['trade_date'], axis=1)
    frame = frame.drop(['stock_code'], axis=1)
    frame = frame.drop(['industry_type'], axis=1)
    # frame = frame.drop(['index'], axis=1)
    frame = frame.drop(['closed_price'], axis=1)
    frame = shuffle(frame)
    Y = frame[['label']]
    Y.loc[Y['label'] == -1] = 0
    X = frame.drop('label', 1, inplace=True)
    X = frame.drop('returns', 1, inplace=True)
    # print(frame)
    # the following steps are constructed X_TRAIN, Y_TRAIN, X_TEST, Y_TEST
    # 　print(X_)
    x_train, x_test, y_train, y_test = train_test_split(frame, Y, test_size=0.25, random_state=1)

    num_round = 100
    bst = XGBClassifier(max_depth=2, learning_rate=0.1, n_estimators=num_round, silent=True,
                        objective='binary:logistic')
    eval_set = [(x_train, y_train), (x_test, y_test)]
    bst.fit(x_train, y_train, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True)

    # make prediction
    preds = bst.predict(x_test)
    test_accuracy = accuracy_score(y_test, preds)
    print("Test Accuracy: %.2f%%" % (test_accuracy * 100.0))


if __name__ == '__main__':
    main3()