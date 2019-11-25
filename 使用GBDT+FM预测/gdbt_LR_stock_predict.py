import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os


'''
we can use twice gdbt methods to filter the extreme data and the ordinary data, then we can judge by positive data
and negative data. 
'''

# global variables
STOCK_NUM = 5
T = 72 #the training time is defined as 72 months
whole_list =[]
def gbdt(X,Y,x_test,stock_list,f2):
    # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=1)  # 训练集和测试集
    x_train, x_train_lr, y_train, y_train_lr = train_test_split(X, Y, test_size=0.25)
    grd = GradientBoostingClassifier(n_estimators=310, random_state=10, subsample=0.6, max_depth=7,
                                      min_samples_split=900)
    # 调用one-hot编码。
    grd_enc = OneHotEncoder()
    # 调用LR分类模型。
    grd_lm = LogisticRegression()
    '''使用X_train训练GBDT模型，后面用此模型构造特征'''
    grd.fit(x_train, y_train)
    # fit one-hot编码器
    grd_enc.fit(grd.apply(x_train)[:, :, 0])
    '''
    使用训练好的GBDT模型构建特征，然后将特征经过one-hot编码作为新的特征输入到LR模型训练。
    '''
    grd_lm.fit(grd_enc.transform(grd.apply(x_train_lr)[:, :, 0]), y_train_lr)
    # 用训练好的LR模型多X_test做预测
    y_pred_grd_lm = grd_lm.predict_proba(grd_enc.transform(grd.apply(x_test)[:, :, 0]))[:, 1]
    # 根据预测结果输出
    y_predict = y_pred_grd_lm.tolist()
    dic = dict(zip(stock_list,y_predict))
    dic2 = sorted(dic.items(), key=lambda d: d[1], reverse=True)
    temp_list = []
    for k in dic2:
        if k[1] > 0.99:
            temp_list.append(k[0])
        else:
            break
    # print(temp_list)
    whole_list.append(temp_list[0:5])

def main():
    # we decide to choose the stock set by using GDBT-LR model
    # and the training time is 2010/1/31-2015/12/31 the simulation time is 2016/1/31-2016/12/31
    # the stock related in hs300
    path2 = "./result.txt"
    if not os.path.exists(path2):
        with open(path2, "w") as f1:
            print('the file is already set up')
    with open(path2, "a", encoding='utf-8') as f2:
        f2.write('[')
        for i in range(12):
            path = 'data/'
            frame = pd.DataFrame()
            list_ = []
            for j in range(i+1, i+T+1):
                # print(i)
                # print(j)
                df = pd.read_csv(path + str(j) + ".csv", index_col=0, header=0)
                list_.append(df)
            frame = pd.concat(list_)
            frame = frame.drop(['trade_date'], axis=1)
            frame = frame.drop(['industry_type'], axis=1)
            # frame = frame.drop(['index'], axis=1)
            frame = frame.drop(['closed_price'], axis=1)
            frame = frame.drop(['returns'], axis=1)
            frame = shuffle(frame)
            frame = frame.drop(['stock_code'], axis=1)
            Y = frame[['label']]
            Y.loc[Y['label'] == -1] = 0
            X = frame.drop('label', 1)
            # print(frame)
            # the following steps are constructed X_TRAIN, Y_TRAIN, X_TEST, Y_TEST
            # 　print(X_)

            # now construct a x_test
            try:
                frame2 = pd.read_csv(path + str(i+T+1) + ".csv", index_col=0, header=0)
            except:
                break
            frame2 = shuffle(frame2)
            frame2 = frame2.drop(['trade_date'], axis=1)
            stock_list = frame2['stock_code'].values.tolist()
            frame2 = frame2.drop(['stock_code'], axis=1)
            frame2 = frame2.drop(['industry_type'], axis=1)
            # frame2 = frame2.drop(['index'], axis=1)
            frame2 = frame2.drop(['closed_price'], axis=1)
            frame2 = frame2.drop(['returns'], axis=1)
            X2 = frame2.drop('label', 1)
            try:
                gbdt(X,Y,X2,stock_list,f2)
            except:
                pass
            finally:
                print(whole_list)




if __name__ == '__main__':
    main()