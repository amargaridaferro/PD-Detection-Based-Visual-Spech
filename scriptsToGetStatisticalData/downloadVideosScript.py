# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os

def download(file):
    f = open(file, "r")
    for line in f:
        print("Download de: ", line)
        downloadVideo = "youtube-dl " + line
        os.system(downloadVideo)
    f.close()


if __name__ == '__main__':
    download("")

