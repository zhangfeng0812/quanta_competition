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





def get_info(stock_id):
    db = pymysql.connect(host='192.168.1.252', user='zhangfeng', password='ai436436',
                         port=3306, database='stock_pre_rehabilitation')
    cursor = db.cursor()
    sql = "select updated_time,closed_price from s" + str(stock_id)
    try:
        cursor.execute(sql)
        data = cursor.fetchall()
        return data
    except Exception:
        raise Exception('select info from MySQL failed.')
    finally:
        db.close()


def cal_returns(df):
    returns_lst = []
    month_days_list = df['month_days_closed_price'].tolist()
    one_day_list = df['closed_price'].tolist()
    for i in range(len(df)):
        try:
            returns_lst.append((float(month_days_list[i]) - float(one_day_list[i])) / float(one_day_list[i]))
        except Exception:
            returns_lst.append(0)
            print('special processing')
    df.insert(df.shape[1], 'returns', returns_lst)


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
        df_new.sort_values(by='returns', ascending=False, inplace=True)
        # define the limit is 30 percent stocks
        # drop table contains NAN
        df2 = df_new.dropna(axis=0, how='any')
        ratio = 0.1
        # threshold shorten to 0.2 and the returns is limited to 7% for extreme value if necessary
        for j in range(len(df2)):
            if j < int(ratio*len(df2)):
                labels_lst.append(1)
            elif j > int((1-ratio)*len(df2)):
                labels_lst.append(0)
            else:
                labels_lst.append(-1)
        df2.insert(df2.shape[1], 'label', labels_lst)
        # print(df_new)
        try:
            df2[df2['label'].isin([0, 1])] .to_csv('C:/Users/Administrator/ai436/AI_5_1/data_train/' + str(time3) + '.csv', index=False)
        except Exception:
            logging.error('csv write failed')


def data_process(file_name):
    tmp_lst = []
    with open(file_name, 'r', encoding='UTF-8') as f:
        reader = csv.reader(f)
        for row in reader:
            tmp_lst.append(row)
    df = pd.DataFrame(tmp_lst[1:], columns=tmp_lst[0])
    # 对data frame 进行新增列：当天交易价格。五天后的交易日价格。收益率。标签。
    # initialize
    '''
    df.insert(df.shape[1], 'closed_price', -1)
    df.insert(df.shape[1], 'month_days_closed_price', -1)
    df.insert(df.shape[1], 'returns', -1)
    df.insert(df.shape[1], 'label', -1)
    '''
    closed_price_list = []
    month_days_closed_price_list = []
    # Traverse the whole data frame
    cnt = 0  # 为了定位，减少程序运行时间
    price = get_info('000001')
    stock_set = set()
    for i in range(len(df)):
        cnt += 1
        temp_trade_day = time.strptime(df.iloc[i,1], "%Y/%m/%d")
        trade_day = time.strftime("%Y-%m-%d", temp_trade_day)
        stock = df.iloc[i]['code'][:6]
        # 先简单处理 暂时到2019-10月
        if trade_day.startswith('2019-10'):
            closed_price_list.append(np.NAN)
            month_days_closed_price_list.append(np.NAN)
            continue
        if stock not in stock_set:
            price = get_info(stock)
            stock_set.add(stock)
        # 为了排除个股退市
        if price == ():
            closed_price_list.append(np.NAN)
            month_days_closed_price_list.append(np.NAN)
            continue
        # insert the data into data frame
        try:
            price[cnt][0]
        except Exception:
            cnt = 0
        # 只是为了解决数据越界错误
        if price[cnt][0] == trade_day:
            closed_price_list.append(price[cnt][1])
            month_days_closed_price_list.append(price[cnt+20][1])
        else:
            for j in range(len(price)):
                if price[j][0] == trade_day:
                    cnt = j
                    closed_price_list.append(price[cnt][1])
                    month_days_closed_price_list.append(price[cnt + 20][1])
                    break
    df.insert(df.shape[1], 'closed_price', closed_price_list)
    df.insert(df.shape[1], 'month_days_closed_price', month_days_closed_price_list)
    df = df.dropna(axis=0, how='any')
    cal_returns(df)
    split_df(df)



def training_data_gen():
    try:
        os.mkdir('C:/Users/Administrator/ai436/AI_5_1/data_train')
    except Exception:
        pass
    # data_process just for training data
    # data_process('DFF.csv')
    date_list = [ '2013.csv', '2014.csv', '2015.csv', '2016.csv', '2017.csv',
                 '2018.csv', '2019.csv']
    for date in date_list:
        data_process(date)
















if __name__ == '__main__':
    training_data_gen()
