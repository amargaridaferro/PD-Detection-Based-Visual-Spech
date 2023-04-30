import os
import csv

def count_files(file):
    with open(file, 'r') as f:

        for line in f:
            lineSplited = line.split(",")
            videoID = lineSplited[1]
            channelID = lineSplited[2]
            speakerID = lineSplited[8]

            path = '../Tese/PD/video_preprocess/APD/segmented_videos/' + speakerID
            files_list = os.listdir(path)  # dir is your directory path
            number_files = len(files_list)

            row = channelID+','+videoID+','+speakerID+','+str(number_files)
            print(row)
            
    f.close()


if __name__ == '__main__':
    count_files('../APD.csv')
