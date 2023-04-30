import os
import csv
from datetime import datetime


def total_seconds(file):

    with open(file, 'r') as f:

        for line in f:
            newLine = line.rstrip()
            newLine+=',end'
            print(newLine)
    f.close()


if __name__ == '__main__':
    total_seconds('../AHC.csv')

