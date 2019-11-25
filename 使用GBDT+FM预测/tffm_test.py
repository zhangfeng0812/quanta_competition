import tensorflow as tf
from tffm import TFFMClassifier
from AI5_1.data_process import get_last_trade_day
import logging
import method.logging_file
from sklearn.utils import shuffle
import numpy as np
from tqdm import tqdm
import pandas as pd
T = 60



def read_stock(time_now):
    root_dir = predict_data_path+str(time_now)+"stock.txt"
    lines = []
    with open(root_dir, 'r') as file_to_read:
        while True:
            line = file_to_read.readline()
            if not line:
                break
            line = line.strip('\n')
            lines.append(line)
    return lines



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



whole_list = []
predict_data_path = "E:/factor_data/output_fm_prob/"
method.logging_file.log_file('tffm')
test_data_path = "E:/factor_data/month_test_data/"
train_data_path = "E:/factor_data/month_train_data/"
time_list = get_last_trade_day('2010-01-01', '2019-09-30')
for i in range(len(time_list)-T):
    X_tr, y_tr = get_training_set(i)
    X_te, stock_list = get_test_set(i)
    X_tr = X_tr.values
    y_tr = y_tr.values
    y_test_label = np.array(y_tr)
    X_test_label = list(map(int, y_test_label))
    X_test_label = np.transpose(X_test_label)
    X_te = X_te.values
    model = TFFMClassifier(
        order=2,
        rank=10,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
        n_epochs=50,
        batch_size=1024,
        init_std=0.001,
        reg=0.01,
        input_type='dense',
        seed=42
    )
    model.fit(X_tr, X_test_label, show_progress=True)
    predictions = model.predict_proba(X_te)[:, 1]
    dic = dict(zip(stock_list, predictions))
    dic2 = sorted(dic.items(), key=lambda d: d[1], reverse=True)
    temp = []
    for j in range(5):
        temp.append(dic2[j][0])
    whole_list.append(temp[0:5])
    print(whole_list)
    # this will close tf.Session and free resources
    model.destroy()

print(whole_list)