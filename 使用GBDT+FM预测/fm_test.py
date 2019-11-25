import xlearn as xl
from AI5_1.data_process import get_last_trade_day
import method.logging_file
import logging


def read_output(time_now):
    root_dir = predict_data_path+str(time_now)+".txt"
    lines = []
    with open(root_dir, 'r') as file_to_read:
        while True:
            line = file_to_read.readline()
            if not line:
                break
            line = line.strip('\n')
            lines.append(line)
    return lines

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

def fm_predict(time_now):
    fm_model = xl.create_fm()  # Use factorization machine
    fm_model.setTrain(train_data_path+str(time_now)+".txt")  # Training data
    # fm_model.setValidate(test_data_path+str(time_now)+".txt")  # Set the path of validation dataset
    # param:
    #  0. Binary classification task
    #  1. learning rate: 0.2
    #  2. lambda: 0.002
    #  3. metric: accuracy
    param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric':'acc', 'k':3}
    fm_model.fit(param, output_data_path+time_now+".out")
    # Use cross-validation
    fm_model.setTest(test_data_path+str(time_now)+".txt")  # Set the path of test dataset
    fm_model.setSigmoid()
    fm_model.predict(output_data_path+time_now+".out", predict_data_path+str(time_now)+".txt")



if __name__ == '__main__':
    method.logging_file.log_file('fm_test')
    stock_list = []
    T = 60
    # get the whole trade date in the last day of the month
    test_data_path = "E:/factor_data/small_test/"
    train_data_path = "E:/factor_data/small_train/"
    output_data_path = "E:/factor_data/output_fm/"
    predict_data_path = "E:/factor_data/output_fm_prob/"
    time_list = get_last_trade_day('2010-01-01', '2019-10-31')
    for i in range(len(time_list)-T):
        time_now = time_list[i]
        try:
            fm_predict(time_now)
            score = read_output(time_now)
            stockList = read_stock(time_now)
            dic = dict(zip(stockList, score))
            dic2 = sorted(dic.items(), key=lambda d: d[1], reverse=True)
            temp = []
            for j in range(10):
                temp.append(dic2[j][0])
            stock_list.append(temp)

        except Exception:
            pass
        finally:
            print(stock_list)
