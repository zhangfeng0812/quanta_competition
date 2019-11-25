import pymysql
import datetime
import pandas as pd
import csv
import time
import logging
import os
import numpy as np
'''
以月度为单位
'''


def get_last_day(my_time):
    my_day = datetime.datetime(int(my_time[0:4]), int(my_time[5:7]), int(my_time[8:10]))
    # now = datetime.datetime.now()
    delta = datetime.timedelta(days=-1)
    my_yesterday = my_day + delta
    my_yes_time = my_yesterday.strftime('%Y-%m-%d')
    return my_yes_time


def get_last_trade_day(begin_time, end_time):

    def get_date(beginDate, endDate):
        date_index = pd.date_range(beginDate, endDate)
        days = [pd.Timestamp(x).strftime("%Y-%m-%d") for x in date_index.values]
        tmp = []
        for index, v in enumerate(days):
            if index == len(days) - 1:
                tmp.append(days[index])
            if index == 0:
                tmp.append(days[0])
            else:
                _ = v.split('-')[2]
                if _ == '01':
                    tmp.append(days[index - 1])
                    tmp.append(days[index])
        last_trade_days = []
        for i in range(len(tmp) // 2):
            last_trade_days.append(tmp[i * 2 + 1])
        return last_trade_days

    db = pymysql.connect(host='', user='', password='',
                         port=3306, database='stock_init')
    cursor = db.cursor()
    sql = "select trade_day from trade_days"
    try:
        cursor.execute(sql)
        data = cursor.fetchall()
        temp_lst = data
        time_lst = []

        for temp_time in temp_lst:
            time_lst.append(temp_time[0])
    except Exception:
        raise Exception('select info from MySQL failed.')
    finally:
        db.close()
    last_day = get_date(begin_time,end_time)
    last_trade_day = []
    for day in last_day:
        temp_delta = 10
        while temp_delta:
            temp_delta -= 1
            if day in time_lst:
                last_trade_day.append(day)
                break
            else:
                day = get_last_day(day)
    return last_trade_day


def read_csv_test_data(file_name):

    def change_time_format_in_df(df):
        try:
            unsolved_list = df['date'].tolist()
        except Exception:
            unsolved_list = df['date'].iloc[:,1].tolist()
        solved_list = []
        for unsolved_date in unsolved_list:
            time2 = time.strptime(unsolved_date, "%Y/%m/%d")
            solved_list.append(time.strftime("%Y-%m-%d", time2))
        df2 = df.drop(['date'], axis=1)
        df2.insert(1,'date',solved_list)
        return df2

    def split_df(df,year_trade_day):
        new_df = change_time_format_in_df(df)
        for last_day in year_trade_day:
            df_new = new_df[new_df['date'].isin([last_day])]
            # define the limit is 30 percent stocks
            # drop table contains NAN
            df2 = df_new.dropna(axis=0, how='any')
            # print(df_new)
            try:
                df2.to_csv(test_data_path+str(last_day)+'.csv', index=False)
            except Exception:
                logging.error('csv write failed')

    tmp_lst = []
    with open(data_path+file_name, 'r', encoding='UTF-8') as f:
        reader = csv.reader(f)
        for row in reader:
            tmp_lst.append(row)
    df = pd.DataFrame(tmp_lst[1:], columns=tmp_lst[0])
    year_trade_day = [x for x in time_list if x.startswith(file_name[0:4])]  # 该年每个月的最后一个交易日
    # 先进行筛选分表格的过程
    split_df(df,year_trade_day)


def read_csv_training_data():

    def get_info(stock_id,cnt):
        db = pymysql.connect(host='', user='', password='',
                             port=3306, database='stock_pre_rehabilitation')
        cursor = db.cursor()
        sql = "select closed_price from s" + str(stock_id)+" where updated_time like '"+str(time_list[cnt])+"'"
        try:
            cursor.execute(sql)
            data = cursor.fetchone()
            if data is None:
                return np.NaN
            return data[0]
        except Exception:
            raise Exception('select info from MySQL failed.')
        finally:
            db.close()

    def cal_returns(df):
        returns_lst = []
        month_days_list = df['next_months_closed_price'].tolist()
        one_day_list = df['closed_price'].tolist()
        for i in range(len(df)):
            try:
                returns_lst.append((float(month_days_list[i]) - float(one_day_list[i])) / float(one_day_list[i]))
            except Exception:
                returns_lst.append(0)
                print('special processing')
        df.insert(df.shape[1], 'returns', returns_lst)

    def split_df(df):
        labels_lst = []
        df.sort_values(by='returns', ascending=False, inplace=True)
        df2 = df.dropna(axis=0, how='any')
        ratio = 0.3
        # threshold shorten to 0.2 and the returns is limited to 7% for extreme value if necessary
        for j in range(len(df2)):
            if j < int(ratio * len(df2)):
                labels_lst.append(1)
            elif j > int((1 - ratio) * len(df2)):
                labels_lst.append(0)
            else:
                labels_lst.append(-1)
        df2.insert(df2.shape[1], 'label', labels_lst)
        # print(df_new)
        try:
            df2[df2['label'].isin([0, 1])].to_csv(train_data_path + str(time_list[cnt]) + '.csv', index=False)
        except Exception:
            logging.error('csv write failed')

    for cnt in range(len(time_list)):
        flag = False
        if cnt == len(time_list)-1:
            flag = True
            time_list.append('2019-10-31')
        tmp_lst = []
        with open(test_data_path + time_list[cnt]+".csv", 'r', encoding='UTF-8') as f:
            reader = csv.reader(f)
            for row in reader:
                tmp_lst.append(row)
        df = pd.DataFrame(tmp_lst[1:], columns=tmp_lst[0])
        # 对data frame 进行新增列：当天交易价格。五天后的交易日价格。收益率。标签。
        closed_price_list = []
        next_months_closed_price_list = []
        for i in range(len(df)):
            stock = df.iloc[i]['code'][:6]
            closed_price_list.append(str(get_info(stock, cnt)))
            next_months_closed_price_list.append(get_info(stock, cnt+1))
        df.insert(df.shape[1], 'closed_price', closed_price_list)
        df.insert(df.shape[1], 'next_months_closed_price', next_months_closed_price_list)
        df = df.dropna(axis=0, how='any')
        cal_returns(df)
        # 进行筛选分表格的过程
        split_df(df)




        if flag:
            time_list.pop()

if __name__ == '__main__':
    # 获取所有月末的交易日
    time_list = get_last_trade_day('2010-01-01', '2019-09-30')
    data_path = "E:/factor_data/"
    test_data_path = "E:/factor_data/month_test_data/"
    train_data_path = "E:/factor_data/month_train_data/"
    try:
        os.mkdir(data_path+"month_train_data")
    except Exception:
        pass
    '''
    for year in ['2010.csv', '2011.csv', '2012.csv', '2013.csv', '2014.csv', '2015.csv', '2016.csv', '2017.csv',
                 '2018.csv', '2019.csv']:
        read_csv_test_data(year)
    '''
    read_csv_training_data()