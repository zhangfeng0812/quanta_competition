


def read_output(i):
    root_dir = "C:/Users/Administrator/ai436/ML_Essay/output"+str(i)+".txt"
    lines = []
    with open(root_dir, 'r') as file_to_read:
        while True:
            line = file_to_read.readline()
            if not line:
                break
            line = line.strip('\n')
            lines.append(line)
    return lines

def read_stock(i):
    root_dir = "C:/Users/Administrator/ai436/ML_Essay/stock_list"+str(i)+".txt"
    lines = []
    with open(root_dir, 'r') as file_to_read:
        while True:
            line = file_to_read.readline()
            if not line:
                break
            line = line.strip('\n')
            lines.append(line)
    return lines


if __name__=='__main__':
    stock = []
    for i in range(1,24):
        stock_list = []
        score = read_output(i)
        stockList = read_stock(i)
        print(len(score))
        print(len(stockList))
        dic = dict(zip(stockList,score))
        dic2 = sorted(dic.items(), key=lambda d: d[1], reverse=True)
        for j in range(5):
            stock_list.append(dic2[j][0])
        stock.append(stock_list)
    print(stock)
