# April 2023, Ana Margarida Ferro

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

