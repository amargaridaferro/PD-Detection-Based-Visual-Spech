# April 2023, worked based on Catarina Botelho
# Ana Margarida Ferro

import re
import sys
import os
import argparse
import subprocess
import numpy as np
import datetime
import time

def argparser():
    parser = argparse.ArgumentParser(description='Process captions file.')
    parser.add_argument('--vid','-v', required=True,help='.wav video file')
    parser.add_argument('--rt','-t', required=True,help='Transcription file')
    parser.add_argument('--poi', required=True,help='POI ID')
    parser.add_argument('--outputdir', required=True, help='directory for saving segmented videos')
    parser.add_argument('--datainfodir', required=True, help='directory for saving data info, such as scp, utt2spk...')
    parser.add_argument('--counter',default=0,help='Utterance counter (for more videos)')
    parser.add_argument('--min_dur',default=0,help='Minimum duration for each segment, in seconds.')
    parser.add_argument('--max_dur',default=9999,help='Maximum duration for each segment, in seconds.')
    parser.add_argument('--optimal_dur',default=0,help='Optimal duration for each segment, in seconds.')

    return parser.parse_args()


def segment_video(video, beg_t, end_t, out_vid_file, spk, utt, utt2spk_file, scp_file):
    try:
        from subprocess import DEVNULL
    except ImportError: # Python 2
        DEVNULL = open(os.devnull, 'r+b', 0)
            
    with open(scp_file,'a') as scp:
        scp.write(utt + ' ' + out_vid_file + '\n')
    with open(utt2spk_file,'a') as utt2spk:
        utt2spk.write(utt + ' ' + spk + '\n')
    subprocess.check_call(['ffmpeg', '-y', '-i', video,'-ss', beg_t, '-to', end_t, out_vid_file], stdin=DEVNULL) #.wait()

        

def time_to_sec(time_string):
    seconds = time.strptime(time_string.split('.')[0],'%H:%M:%S')
    seconds = datetime.timedelta(hours=seconds.tm_hour,minutes=seconds.tm_min,seconds=seconds.tm_sec,milliseconds=float(time_string.split('.')[1])).total_seconds()
    return seconds


def sec_to_time(secs):
    return str(datetime.timedelta(seconds=secs))


def main():

    args = argparser()

    # frame counter
    i=args.counter
    l=0


    spk = args.poi
    scp_file = args.datainfodir + '/' + spk + '/wav.scp'
    text_file = args.datainfodir + '/' + spk + '/text'
    utt2spk_file = args.datainfodir + '/' + spk + '/utt2spk'

    subprocess.call(["mkdir","-p",args.outputdir + '/' + spk])
    subprocess.call(["mkdir","-p",args.datainfodir + '/' + spk])

    with open(args.rt,'r') as rt:

        for line in rt:
            # start boolean representing sucessful spliting of the video to False
            sucess_bool = True
            
            # read times from line
            times = re.findall('\d\d:[0-5]\d:[0-5]\d.\d\d\d',line)

            # if this is a times line
            if len(times):

                beg_t = times[0]
                end_t = times[1]

                
                # convert times to seconds
                beg_t_sec = time_to_sec(beg_t)
                end_t_sec = time_to_sec(end_t)
                interval = end_t_sec - beg_t_sec


                if interval > args.min_dur and interval < args.max_dur:
                    utt = spk + '-' + str(l).zfill(5)  + '-' + str(i).zfill(5)
                    out_vid_file = args.outputdir + '/' + spk + '/' + utt + '.wav'
                    
                    segment_video(args.vid, beg_t, end_t, out_vid_file, spk, utt, utt2spk_file, scp_file)
                    i+=1
                    l+=1
                
                elif interval > args.max_dur:
                    b=beg_t_sec
                    for e in np.arange(args.opt_dur, interval, args.opt_dur):
                        utt = spk + '-' + str(l).zfill(5)  + '-' + str(i).zfill(5)
                        out_vid_file = args.outputdir + '/' + spk + '/' + utt + '.wav'
                        
                        beg_t_segm = sec_to_time(b)
                        end_t_segm = sec_to_time(b + e)
                        segment_video(args.vid, beg_t_segm, end_t_segm, out_vid_file, spk, utt, utt2spk_file, scp_file)
                        i+=1
                        b = b + e
                    l+=1
                else:
                    sucess_bool=False

            # transcription line
            elif len(line.split()):
                if sucess_bool:
                    whole_utt = spk + '-' + str(l).zfill(5) 
                    with open(text_file,'a') as txt:
                        txt.write(whole_utt + ' ' + line)


    print("Finished segment_videos script for subject ", spk)


if __name__ == "__main__":
        main()
