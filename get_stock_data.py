#url: https://query1.finance.yahoo.com/v7/finance/download/<STOCK_NAME>?period1=<UNIX_TIME_BEGIN>&period2=<UNIX_TIME_END>&interval=1d&events=history&crumb=
from enum import Enum
from csv import reader


Open = 0
High = 1
Low = 2
Close = 3
Adj = 4
Volume = 6


def get_data(file_name):
    with open(file_name, "r") as csvf:
        csv_reader = reader(csvf)
        csv_reader.next()
        data = [record for record in csv_reader]
        for record in data:
            # remove date field
            record.pop(0)
            for index, value in enumerate(record):
                record[index] = float(value)
        return data


def split_data_to_intervals(interval_length, data):
    return [data[i:i+interval_length] for i in range(0, len(data))]


"""
generate in the given interval length from the given data
test is defined as <prediction_interval> consecutive records and the record of the <prediction_interval>+1-th day
"""
def generate_prediction_tests(prediction_interval, data):
    intervals = split_data_to_intervals(prediction_interval + 1, data)
    tests = []
    for interval in intervals:
        interval_len = len(interval)
        test = (interval[:interval_len - 1], interval[interval_len - 1][Close])
        tests.append(test)
    return tests



def main():
    data = get_data(r"c:\test\TEVA.csv")
    tests = generate_prediction_tests(5, data)
    print tests

if __name__ == '__main__':
    main()
