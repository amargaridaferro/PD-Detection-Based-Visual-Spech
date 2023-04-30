#!/bin/bash

#########################################################################
#                                                                       #
# Script for download YT videos and transcriptions                      #
# --------------------------------------------------------------------- #
# Date: June 2022        based on Catarina Botelho work                                              #
#                                                                       #
#########################################################################

# Config start to install ubunto (wsl version 2 is mandatory for the last step)

#sudo apt install python3-pip
#pip install numpy
#pip install tqdm
#pip install torch
#pip install opencv-python
#pip install librosa



video_info_path=$1 #"videos_PD_partition_.csv"  # this file has the header:
                    # "yt_id,channel,wsm_keyword,diagnosis,gender,age,speaker_id,ti,tf,"
                    # the last comma is necessary because of the /n at the end
root="Tese/PD"
video_preproces_dir=$root/"video_preprocess"
pre_process_scripts_dir=$root/"pre_process_scripts"
lip_reading_dir=$root/"lip_reading/Lipreading_using_Temporal_Convolutional_Networks-master"
cuda=0

# Stages:
STEP0=0  # downloads yt videos and trasncrptions DONE

STEP1=0 # remove frames where there is no speech DONE

STEP2=0 # cut subitles to the current and get new videos's duration

STEP3=0 # one second segmentation - videos

STEP4=0 # prepare data.csv with data info: colapse data information to single file

STEP5=0 # Dedices whether to exclude or keep a video segment based on face detection. Extracts facial landmarks for all accepted video segments.

STEP6=0 # Crops each video to mouth rois, using landmarks. Coverts to gray

STEP7=1   # Extracts lip reading embeddings. Updates data.csv



if [ $STEP0 -eq "1" ]; then
  echo "----------- Starting step 0. ----------- "

  # creating necessary folders
  #mkdir -p ${video_preproces_dir}/raw_transcriptions/
  #mkdir -p ${video_preproces_dir}/raw_videos/
  #mkdir -p ${video_preproces_dir}/processed_transcriptions/
  #mkdir -p ${video_preproces_dir}/segmented_videos/

  while IFS=, read -r keyword videoID channelID diagnosis gender age role category speakerID ti tf c
  do

      echo  "Video $videoID from speaker $speakerID with ti=$ti and tf=$tf."

      yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]" https://www.youtube.com/watch?v=${videoID} -o ${video_preproces_dir}/yt_videos/'/%(id)s.%(ext)s'
      yt-dlp --skip-download --write-auto-subs --sub-lang en https://www.youtube.com/watch?v=${videoID} -o ${video_preproces_dir}/yt_transcriptions/'/%(id)s.%(ext)s'
      ffmpeg -nostdin -i ${video_preproces_dir}/yt_videos/${videoID}.mp4 -ss $ti -to $tf -async -1 ${video_preproces_dir}/raw_videos/${speakerID}.mp4

    done < ${video_info_path}
fi


if [ $STEP1 -eq "1" ]; then
  echo "----------- Starting step 1. ----------- "
  # The goal of VAD is to remove frames where there is no speech. This is an important step as we don't want to
  # feed non-speech data to our models. We assume that the captions we get already do this processing step.
  while IFS=, read -r keyword videoID channelID diagnosis gender age role category speakerID ti tf totalTs
  do
    if test -f "${video_preproces_dir}/yt_transcriptions/${videoID}.en.vtt"; then
      echo "Processing $speakerID's transcription."

      # check if file type is really vtt, or src
      is_vtt=`grep "</c>" ${video_preproces_dir}/yt_transcriptions/${videoID}.en.vtt | wc -l`

      # process captions
      python3 ${pre_process_scripts_dir}/utils/process_captions.py -t 'vtt' -f ${video_preproces_dir}/yt_transcriptions/${videoID}.en.vtt > ${video_preproces_dir}/processed_transcriptions/${speakerID}.rt
    else
      "Missing transcription for $speakerID."
    fi
  done < ${video_info_path}
fi


if [ $STEP2 -eq "1" ]; then
  mkdir -p ${video_preproces_dir}/cut_transcriptions/

  echo "----------- Starting step 2. ----------- "
  while IFS=, read -r keyword videoID channelID diagnosis gender age role category speakerID ti tf totalTs
  do
    # Segment videos transcription for the specific range and obtain new duration
    python3 ${pre_process_scripts_dir}/utils/cut_video_subs.py --ti $ti --tf $tf --rt ${video_preproces_dir}/processed_transcriptions/${speakerID}.rt --output ${video_preproces_dir}/cut_transcriptions/${speakerID}.rt
  done < ${video_info_path}
fi


if [ $STEP3 -eq "1" ]; then
  echo "----------- Starting step 3. ----------- "
  while IFS=, read -r keyword videoID channelID diagnosis gender age role category speakerID ti tf totalTs
  do
    cp ${video_preproces_dir}/yt_videos/${videoID}.mp4 ${video_preproces_dir}/raw_videos/${speakerID}.mp4
    echo "Copy completed for $speakerID"

    # Segment videos in one second segmentation
    python3 ${pre_process_scripts_dir}/utils/segment_videos_w_trasncription.py --vid ${video_preproces_dir}/raw_videos/${speakerID}.mp4 \
                              --rt ${video_preproces_dir}/cut_transcriptions/${speakerID}.rt \
                              --poi ${speakerID} \
                              --outputdir ${video_preproces_dir}/segmented_videos/ \
                              --datainfodir ${video_preproces_dir}/segmented_videos_info/
  done < ${video_info_path}
fi



if [ $STEP4 -eq "1" ]; then
  echo "----------- Starting step 4. ----------- "
  mkdir -p ${root}/data_info/

  # necessary to add header to APD.csv. To run this step we need to do sh video_preprocess.sh APD.csv and with the headers as
  # make data csv file suggests

  python3 ${pre_process_scripts_dir}/utils/make_data_csv.py \
                           --original_video_info_csv ${video_info_path} \
                           --segm_video_info_dir ${video_preproces_dir}/segmented_videos_info/ \
                           --output_csv ${root}/data_info/dataPD.csv
fi



if [ $STEP5 -eq "1" ]; then
  echo "----------- Starting step 5. ----------- "
  mkdir -p ${root}/features/landmarks/

  # Select videos based on face detection. Extract facial landmarks
  python3 ${pre_process_scripts_dir}/utils/detect_faces_video.py \
                                --data_info ${root}/data_info/dataPD.csv \
                                --outputdir ${root}/features/landmarks/
fi



if [ $STEP6 -eq "1" ]; then
  echo "----------- Starting step 6. ----------- "

  mkdir -p ${root}/data/mouth_rois/

  # Crop mouth ROIs
  python3 ${pre_process_scripts_dir}/lip_readings/crop_mouth_from_video.py \
      --video-direc ${video_preproces_dir}/segmented_videos \
      --landmark-direc ${root}/features/landmarks/ \
      --filename-path ${root}/data_info/dataPD_only_faces.csv \
      --save-direc ${root}/data/mouth_rois/ --convert-gray
fi


#STEP 7
#pip3 install torch
#pip3 install tqdm

if [ $STEP7 -eq "1" ]; then
  echo "----------- Starting step 7. ----------- "

  {
  read
  while read line; do
    uid=`echo -n "$line" | cut -d "," -f 1`
    subject=`echo -n "$line" | cut -d "," -f 2`
    if test -f "${root}/features/lipread_emb/$subject/$uid.npz"; then
        echo "File ${root}/features/lipread_emb/$subject/$uid.npz  exists. Skipping it."
    else
        echo "Extracting lip reading embeddings: subject $subject utterance $uid"

        mkdir -p ${root}/features/lipread_emb/$subject/

        CUDA_VISIBLE_DEVICES=$cuda python3 ${lip_reading_dir}/main.py \
            --extract-feats \
            --config-path ${lip_reading_dir}/configs/lrw_resnet18_mstcn.json \
            --model-path ${lip_reading_dir}/models/lrw_resnet18_mstcn_adamw_s3.pth.tar \
            --mouth-patch-path ${root}/data/mouth_rois/${subject}/${uid}.npz \
            --mouth-embedding-out-path ${root}/features/lipread_emb/$subject/$uid.npz
    fi
  done
  } < ${root}/data_info/dataPD_only_faces.csv

  # Update data.csv with roi paths and embedding paths
 awk -F',' -v root=$root 'NR==1 {print $0 ",mouth_roi_path,lipread_emb_path"} NR>1 {print $0 "," root "/data/mouth_rois/" $2 "/" $1 ".npz," root "/features/lipread_emb/"  $2 "/" $1 ".npz,"}' ${root}/data_info/dataPD_only_faces.csv > tmp
 mv tmp ${root}/data_info/dataPD_only_faces.csv


fi


