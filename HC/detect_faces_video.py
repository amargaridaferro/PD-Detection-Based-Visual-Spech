# general packages
import numpy as np
import pandas as pd
import math
import argparse
import time
import subprocess

# computer vision/face processing packages
import cv2
import dlib
import face_recognition
from imutils.video import FileVideoStream

# my functions
from face_utils import face_detect_nn
from face_utils import get_landmarks

"""
Script for....

Adapted from:
https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/
https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/ 
"""

def argparser():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_info", "-d", required=True,
        help="path to data_.csv Header must include uid, subject_id, video_path.")
    ap.add_argument("--faceModel", "-m",
        default="Tese/HC/pre_process_scripts/opencv_dlib/opencv_face_detector_uint8.pb",
        help="path to face detection model.")
    ap.add_argument("--faceConfig", "-cfg",
        default="Tese/HC/pre_process_scripts/opencv_dlib/opencv_face_detector.pbtxt",
        help="path to configuration for face detection model.")
    ap.add_argument("--confidence", "-c", type=float, default=0.9,
        help="minimum probability to filter weak detections")
    ap.add_argument("--min_face_ratio", "-f", type=float, default=0.75,
        help="minimum ratio of frames with faces over total number of frames.")
    ap.add_argument("--landmarkModel", "-l",
        default="Tese/HC/pre_process_scripts/opencv_dlib/shape_predictor_68_face_landmarks.dat",
        help="path to landmarks predictor model.")
    ap.add_argument("--outputdir", "-o", required=False,
        default="Tese/HC/features/landmarks/",
        help="path to directory where to save landmarks.")
    ap.add_argument("--log", required=False,
        default="Tese/HC/features/face_landmarks.log",
        help="path to directory where to save landmarks.")
    args = ap.parse_args()
    return args


def read_video(path):
    vs = FileVideoStream(path, queue_size=16).start() ## uses threading for faster frame reading
    time.sleep(2.0)
    return vs


def video_face_detection(video_path, net, confidence, log):

    # initialize the video stream
    vs = read_video(video_path)
    
    # initialize results list and counters
    frames_w_faces = 0
    n_frames = 0
    images_info = []

    # for each frame in the video:
    while True:
        frame = vs.read()
        if frame is None:
            break
        
        # face detection
        detected_img, face_box = face_detect_nn(frame, net, min_confidence=confidence)  
        n_frames +=1
        
        # if face detected ....
        if len(face_box):
            frames_w_faces += 1
            images_info.append({'resized_img':detected_img, 'face': True,'box': face_box})
        else:
            images_info.append({'resized_img':detected_img, 'face': False})
        
    vs.stop()
    if n_frames:
        return frames_w_faces/n_frames, images_info
    else:
        with open(log, 'a') as f:
            f.write('Could not load video ' + video_path + '. Skipping it.')
        return 0, images_info


def video_landmark_detection(images_info, predictor):

    # for each frame in the video:
    for i, frame_d in enumerate(images_info): 
        
        # check that a face was detected
        if frame_d['face']:
           
            # face detection
            _, shape = get_landmarks(frame_d['resized_img'], frame_d['box'], predictor) 
            frame_d['facial_landmarks'] = shape

    return images_info


def main():

    # load arguments
    args = argparser()
    print(args)
    
    # load models
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromTensorflow(args.faceModel, args.faceConfig)
    landmark_model = dlib.shape_predictor(args.landmarkModel)

    data_df = pd.read_csv(args.data_info)   # header: uid,speakerID,videoID,channelID,full_utt_id,full_utt_text,
                                            # video_path,keyword,diagnosis,gender,age,ti,tf

    has_face = []
    landmark_path = []
    
    n_iter = len(data_df)
    print ("-- Face detection and landmark extraction: --")
    for i, row in data_df.iterrows():

        # printing progress
        print ("Video ", row.uid, "[", str(i+1), " / ", str(n_iter), "]")
        
        # face detection
        faces_detected_percent, images_info = video_face_detection(row.video_path, net, args.confidence, args.log)

        # assert a minimum of frames contains one face AQUI
        if faces_detected_percent >= args.min_face_ratio:
            print("Accept video")

            # extract landamrks for each frame with a face
            images_info = video_landmark_detection(images_info, landmark_model)

        # save landmarks
        images_info = np.array(images_info)
        output_dir = args.outputdir + '/'  + row.speakerID + '/'
        subprocess.call(["mkdir", "-p", output_dir])
        output_file = output_dir + row.uid + '.npy'
        np.save(output_file, images_info, allow_pickle=True)

        has_face.append(faces_detected_percent >= args.min_face_ratio)
        landmark_path.append(output_file)
 
    # Update data_info file
    data_df['has_face'] = has_face
    data_df['landmark_path'] = landmark_path
    data_df.to_csv(args.data_info, index=False)

    data_df = data_df[data_df.has_face == True]
    path = args.data_info.split('.')[0]
    data_df.to_csv(path + "_only_faces.csv", index=False)

    print('Done! .npy file saved to', output_file)


if __name__ == "__main__":
        main()
