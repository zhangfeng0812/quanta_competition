# coding:utf-8

import pymysql
import csv
import numpy as np
import pandas as pd
import time
import datetime
import logging

# global variables
def get_closed_price(trade_day,stock):
    db = pymysql.connect(host='192.168.1.252', user='zhangfeng', password='ai436436',
                         port=3306, database='stock_pre_rehabilitation')
    closed_price = 0
    cursor = db.cursor()
    sql = "select closed_price from "+str(stock)+" where updated_time like '"+str(trade_day)+"'"
    try:
        cursor.execute(sql)
        closed_price = cursor.fetchone()

    except Exception:
        pass
    finally:
        db.close()


    return closed_price

def get_yesterday(mytime):
    myday = datetime.datetime(int(mytime[0:4]),int(mytime[5:7]),int(mytime[8:10]) )
    # now = datetime.datetime.now()
    delta = datetime.timedelta(days=-1)
    my_yesterday = myday + delta
    my_yes_time = my_yesterday.strftime('%Y-%m-%d')
    return my_yes_time

def cal_returns(df):
    returns_lst = []
    for i in range(len(df)-1):
        if df.iloc[i]['stock_code'] != df.iloc[i+1]['stock_code']:
            returns_lst.append(np.NAN)
        elif df.iloc[i]['closed_price'] == 0 or df.iloc[i+1]['closed_price'] == 0 :
            returns_lst.append(np.NAN)
        else:
            returns_lst.append((float(df.iloc[i+1]['closed_price'])-float(df.iloc[i]['closed_price']))/float(df.iloc[i]['closed_price']))
    returns_lst.append(np.nan)

    df.insert(df.shape[1],'returns',returns_lst)



def split_df(df):
    # row num = 84
    cnt = 0
    for i in range(84):
        labels_lst = []
        time1 = df.iloc[i]['trade_date']
        df_new = df[df['trade_date'].isin([time1])]
        df_new.sort_values(by='returns',ascending=False,inplace=True)
        # define the limit is 30 percent stocks
        # drop table contains NAN
        df2 = df_new.dropna(axis=0,how='any')
        ratio = 0.2
        for j in range(len(df2)):
            if j < int(ratio*len(df2)) and df.iloc[j]['returns'] > 0.07:
                labels_lst.append(1)
            elif j > int((1-ratio)*len(df2)) and df.iloc[j]['returns'] <-0.05:
                labels_lst.append(-1)
            else:
                labels_lst.append(0)
        df2.insert(df2.shape[1],'label',labels_lst)
        # print(df_new)
        cnt += 1
        try:
            df2[df2['label'].isin([-1, 1])].to_csv('data/' + str(cnt) + '.csv', index=False)
        except Exception:
            logging.error('csv write failed')


def main():
    tmp_lst = []
    with open('1.csv', 'r',encoding='UTF-8') as f:
        reader = csv.reader(f)
        for row in reader:
            tmp_lst.append(row)
    df = pd.DataFrame(tmp_lst[1:], columns=tmp_lst[0])
    # 新增一列 用来记录该列的收盘价,初始化为0
    # print(df)
    # add a new column to record the monthly closed price
    # building a stock set to avoid the mistake
    # 1 is a good label, 0 is a worse label, -1 just for initializer
    closed_price_lst = []
    for i in range(len(df)):
        temp_trade_day = time.strptime(df.iloc[i]['trade_date'], "%Y/%m/%d  %H:%M:%S")
        trade_day = time.strftime("%Y-%m-%d", temp_trade_day)
        stock = df.iloc[i]['stock_code'][:6]
        #  print(trade_day)
        # print(stock)
        cnt = 30
        # 没考虑停牌的影响 暂时先简单处理
        while cnt > 0:
            cnt -= 1
            closed_price = get_closed_price(trade_day, "s" + stock)
            if closed_price != None and closed_price != 0:
                break
            else:
                trade_day = get_yesterday(trade_day)
        if cnt == 0:
            closed_price = (0,)
        # print(stock)
        closed_price_lst.append(closed_price[0])
        # add a new column to record the returns
    df.insert(df.shape[1],'closed_price',closed_price_lst)
    cal_returns(df)
        # split data frame
        # create new data frame to record the training and dev test
    df2 = split_df(df)
def main2():
    # for test
    tmp_lst = []
    with open('1.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            tmp_lst.append(row)
    df = pd.DataFrame(tmp_lst[1:], columns=tmp_lst[0])
    split_df(df)
if __name__ == '__main__':
    main()