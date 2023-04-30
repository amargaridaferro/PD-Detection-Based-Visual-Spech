import collections
import contextlib
import sys
import wave
import cv2
import numpy as np
import pandas as pd
import argparse
import time
import subprocess

from multiprocessing import Pool
from imutils.video import FileVideoStream
import webrtcvad

"""
Script for segmenting videos based on webrtcvad.
Script adapted from
https://github.com/wiseman/py-webrtcvad/blob/master/example.py
"""

def argparser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--aggressiveness", required=True,
        help=("Integer between 0 and 3. 0 is the least aggressive about"
        "filtering out non-speech, 3 is the most aggressive. "))
    ap.add_argument("--wav_path", 
        help="path to wav file.")
    ap.add_argument("--mp4_path",
        default="path to mp4 file.",
        help="path to configuration for face detection model.")
    ap.add_argument("--min_chunck_size_s", "-min", type=float, default=0.8,
        help="minimum chunck size in seconds")
    ap.add_argument("--max_chunck_size_s", "-max", type=float, default=1.3,
        help="maximum chunck size in seconds.")
    ap.add_argument("--optim_chunck_size_s", "-opt", type=float, default=1,
        help="optimal chunck size in seconds.")
    ap.add_argument("--outputdir", required=False,
        default="/cfs/tmp/mctb/projects/osa_multimodal/data/segmented_videos/vad/",
        help="path to directory where to save landmarks.")
    ap.add_argument("--outputcsv", required=False,
        default="/cfs/tmp/mctb/projects/osa_multimodal/data_info/WOSA/segmented_videos/data.csv",
        help="path to directory where to save landmarks.")
    ap.add_argument('--use_multi_processing', dest='use_multi_processing', 
        action='store_true', help="Whether to paralelize video writing.")
    ap.add_argument('--num_workers', type=int, default=32,  
        help="Number of parallel processes to use.") 
    args = ap.parse_args()
    return args



# video specific functions:
def video_frame_dur(path):
    """ 
    Takes a video path and computes video frame duartion (milliseconds).
    Function adapted from 
    https://learnopencv.com/how-to-find-frame-rate-or-frames-per-second-\
    fps-in-opencv-python-cpp/
    """
    video = cv2.VideoCapture(path)

    # Find OpenCV version
    (major_ver, _, _) = (cv2.__version__).split('.')

    if int(major_ver)  < 3 :
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else :
        fps = video.get(cv2.CAP_PROP_FPS)
        print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    video.release()
    f_dur = int(1000./fps)
    return fps, f_dur



def segment_video(inpath, segment, fps, outpath):
    start_frame = segment[0]
    end_frame = segment[1]

    vs = FileVideoStream(inpath, queue_size=8).start() ## uses threading for faster frame reading
    time.sleep(2.0)
    frames = []

    # for each frame in the video:
    for idx in range(end_frame):
        frame = vs.read()
        if frame is None:
            break
        if idx >= start_frame:    
            frames.append(frame)
    vs.stop()

    # write segment
    write_video(np.array(frames), fps, outpath)


def write_video(frame_array, fps, outpath):
    
    height, width, _ = frame_array[0].shape
    size = (width,height)
    
    out = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for f in frame_array:
        # writing to a image array
        out.write(f)
    out.release()


def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        nsamples = wf.getnframes()
        pcm_data = wf.readframes(nsamples)
        return pcm_data, sample_rate, nsamples


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration, frame_idx):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration
        self.frame_idx = frame_idx


def frame_generator(frame_duration_ms, audio, sample_rate, frame_idx):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    idx=0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration, frame_idx[idx])
        timestamp += duration
        offset += n
        idx += 1


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                #yield b''.join([f.bytes for f in voiced_frames])
                yield [f.frame_idx for f in voiced_frames]
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        #yield b''.join([f.bytes for f in voiced_frames])
        yield [f.frame_idx for f in voiced_frames]


def summary_info(segments_list, basename, output_csv):
    paths = np.array([p for s, p in segments_list])
    uids = np.array([p.split('/')[-1].split('.')[0] for p in paths])
    full_utt_ids = np.array(['-'.join(u.split('-')[:-1]) for u in uids])
    subject_ids = np.array([basename for i in range(len(uids))])
    texts = np.array(["NI" for i in range(len(uids))])
    
    df = pd.DataFrame({'uid': uids, 'subject_id': subject_ids, 'full_utt_id': full_utt_ids, 'full_utt_text': texts, 'video_path': paths})
    df.to_csv(output_csv, index=False)


def main(): 

    args = argparser()

    print ('Reading inputs.')
    audio, sample_rate, nsamples = read_wave(args.wav_path)
    audio_frame_dur_ms = 30  # notice that webrtcvad only works with 10, 20, 30
    nframes = int((nsamples * 1000 / sample_rate) / audio_frame_dur_ms)
    frame_idx = np.arange(nframes)
    fps, video_frame_dur_ms = video_frame_dur(args.mp4_path)
    basename = args.wav_path.split('/')[-1].split('.')[0]
    
    print ('Initialize vad.')
    vad = webrtcvad.Vad(int(args.aggressiveness))
    min_chunck_size = int(args.min_chunck_size_s * fps)
    max_chunck_size = int(args.max_chunck_size_s * fps)
    opt_chunck_size = int(args.optim_chunck_size_s * fps)

    assert min_chunck_size <= opt_chunck_size
    assert opt_chunck_size <= max_chunck_size
    
    print ('Generate audio frames.')
    frames = frame_generator(audio_frame_dur_ms, audio, sample_rate, frame_idx)
    frames = list(frames)

    print ('Compute VAD.')
    segments = vad_collector(sample_rate, audio_frame_dur_ms, 300, vad, frames)
    
    print ('Convert speech segments of audio to video.')
    # each s in segments contains the index of the frames considered to be 
    # speech. We take the first and last to obtain the start and stop marks 
    # of each segment. We multiply by the scale coefficient which converts 
    # audio frame duration to video frame duration. 
    # This trick is required because webrtcvad only works with frame duration
    # 10, 20, 30 ms.
    s_coeff = fps / (1000 / audio_frame_dur_ms)
    segments=list(segments)
    
    segments = [[int(s[0]*s_coeff), int(s[-1]*s_coeff)] for s in segments if len(s)>=2] 

    # define final list of segments, such that each segment has at most 
    # max_chunck_size duration
    print ('Segmenting video according to vad chuncks and opt_chunck_size.')
    subprocess.call(["mkdir", "-p", args.outputdir + '/' + basename + '/']) 
    final_size_segments = []
    for i, segment in enumerate(segments):
        dur = segment[-1] - segment[0]    
        if dur >= min_chunck_size and  dur <= max_chunck_size: 
            utt = basename + '-' + str(i).zfill(5)  + '-00000'
            path = args.outputdir + '/' + basename + '/' + utt + '.mp4'
            final_size_segments.append((segment, path))

        elif dur > max_chunck_size:
            start=segment[0]
            for j, _ in enumerate(np.arange(opt_chunck_size, dur, opt_chunck_size)):   
                utt = basename + '-' + str(i).zfill(5)  + '-' + str(j).zfill(5)
                path = args.outputdir + '/' + basename + '/' + utt + '.mp4'
                s = [start, start+opt_chunck_size]
                start = start+opt_chunck_size
                final_size_segments.append((s, path))

    # read the video and secoment it according to the 
    # segments in final_size_segments
    if args.use_multi_processing: 
        pool = Pool(args.num_workers) 
    
    for (segment, path) in final_size_segments:
        print(' Writing %s' % (path,))
        if args.use_multi_processing:
            pool.apply_async(segment_video, (args.mp4_path, segment, fps, path), error_callback=lambda err: print(err))
        else:
            segment_video(args.mp4_path, segment, fps, path)
    
    if args.use_multi_processing:
        pool.close()
        pool.join()

    # write .csv file with utterance, speaker and video path information
    summary_info(final_size_segments, basename, args.outputcsv)

    # TODO: optimize segment_video function to avoid reading video file 
    # multiple times.

        

if __name__ == '__main__':
    main()
