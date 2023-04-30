import sys
import numpy as np
from pydub import AudioSegment


def main():

    wav_path=sys.argv[1]
    segm_dur=float(sys.argv[2]) # in seconds
    overlap=float(sys.argv[3]) # in seconds
    outDir=sys.argv[4]

    audio = AudioSegment.from_wav(wav_path)
    length = len(audio) # in milliseconds
    segm_dur = int(segm_dur * 1000)
    overlap = int(overlap * 1000)
    shift = int(segm_dur - overlap)

    basename = wav_path.split('/')[-1].split('.')[0]

    for i, start in enumerate(range(0, length, shift)):
        end = start + segm_dur
        if end <= length:
            # split
            chunk = audio[start:end]

            # save to file
            filename = outDir + '/' + basename + '_' + str(i).zfill(5) + '.wav'
            chunk.export(filename, format ="wav")


    print ("Done! Splitted wav into ", i, " segments with ", segm_dur / 1000, " seconds each.")    

if __name__ == '__main__':
    main()
