import os
import csv

def add_description_HC(file):
    index = 10
    with open(file, 'r') as f:

        for line in f:
            lineSplited = line.split(",")
            keyword = lineSplited[0]
            videoID = lineSplited[1]
            channelID = lineSplited[2]
            diagnosis = lineSplited [3]
            gender = lineSplited [4]
            age = lineSplited [5]
            role = lineSplited [6]
            category = lineSplited [7]
            speakerID = 'c_' + str(index)
            index+=1
            ti = lineSplited [8]
            tf = lineSplited [9]
            row = keyword+','+videoID+','+channelID+','+diagnosis+','+gender+','+age+','+role+','+category+','+\
                speakerID+','+ti+','+tf
            print(row, end='')
    f.close()



def add_description_PD(file):
    index = 10
    with open(file, 'r') as f:

        for line in f:
            lineSplited = line.split(",")
            keyword = lineSplited[0]
            videoID = lineSplited[1]
            channelID = lineSplited[2]
            diagnosis = lineSplited [3]
            gender = lineSplited [4]
            age = lineSplited [5]
            role = lineSplited [6]
            category = lineSplited [7]
            speakerID = 'p_' + str(index)
            index+=1
            ti = lineSplited [9]
            tf = lineSplited [10]
            row = keyword+','+videoID+','+channelID+','+diagnosis+','+gender+','+age+','+role+','+category+','+\
                speakerID+',00:'+ti+',00:'+tf
            newLine = row.rstrip()
            newLine+=',end'
            print(newLine)
    f.close()


if __name__ == '__main__':
    #add_description_HC('../AHC.csv')
    add_description_PD('../APD.csv')

