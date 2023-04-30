import os
import csv

def count_gender(file):
    women = 0
    men = 0
    with open(file, 'r') as f:

        for line in f:
            lineSplited = line.split(",")
            
            gender = lineSplited [4]

            if gender == str(2):
                women +=1
            elif gender == str(1):
                men +=1 
        
    print('women: ', women, ' men: ', men)
            
    f.close()


if __name__ == '__main__':
    count_gender('../AHC.csv')
