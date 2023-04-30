import os
import csv
from datetime import datetime


def total_seconds(file):
    result = 0
    a =0;
    with open(file, 'r') as f:

        for line in f:
            lineSplited = line.split(",")
            
            ti = lineSplited [9]
            tiSplited = ti.split(":")
            
            tf = lineSplited [10]
            tfSplited = tf.split(":")

            time1 = datetime(2023, 3, 19, int(tiSplited[0]), int(tiSplited[1]), int(tiSplited[2]))
            time2 = datetime(2023, 3, 19, int(tfSplited[0]), int(tfSplited[1]), int(tfSplited[2]))

            delta = time2 - time1

            # calculate the total number of seconds in the delta
            total_seconds = delta.total_seconds()

            result += total_seconds
            a+=1
            print(total_seconds)

    print(result)
    print(a)

    f.close()


if __name__ == '__main__':
    total_seconds('../APD.csv')

