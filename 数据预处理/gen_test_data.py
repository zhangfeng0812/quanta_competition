import os

# coding:utf-8
import pymysql
import csv
import pandas as pd
import datetime
import time
import numpy as np
import logging
import os


def split_df(df):
    # row num = 84
    flag = True
    time_set = set()
    time1 = df['date'].tolist()

    for temp_time in time1:
        time_set.add(temp_time)
        # time2 = time.strptime(temp_time, "%Y/%m/%d")
        # time3 = time.strftime("%Y-%m-%d", time2)
        # time_set.add(time3)
    for i in time_set:
        time2 = time.strptime(i, "%Y/%m/%d")
        time3 = time.strftime("%Y-%m-%d", time2)
        labels_lst = []
        df_new = df[df['date'].isin([i])]
        # define the limit is 30 percent stocks
        # drop table contains NAN
        df2 = df_new.dropna(axis=0, how='any')
        try:
            df2.to_csv('C:/Users/Administrator/ai436/AI_5_1/data_test/' + str(time3) + '.csv', index=False)
        except Exception:
            logging.error('csv write failed')


def data_process(file_name):
    tmp_lst = []
    with open(file_name, 'r', encoding='UTF-8') as f:
        reader = csv.reader(f)
        for row in reader:
            tmp_lst.append(row)
    df = pd.DataFrame(tmp_lst[1:], columns=tmp_lst[0])
    df = df.dropna(axis=0, how='any')
    split_df(df)



def training_data_gen():
    try:
        os.mkdir('C:/Users/Administrator/ai436/AI_5_1/data_test')
    except Exception:
        pass
    # data_process just for training data
    # data_process('DFF.csv')
    date_list = ['2011.csv', '2014.csv', '2015.csv', '2016.csv', '2017.csv',
                 '2018.csv', '2019.csv']
    for date in date_list:
        data_process(date)
















if __name__ == '__main__':
    training_data_gen()
