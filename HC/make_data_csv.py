#April 2023, based on Catarina Botelho work
#Ana Margarida Ferro
import numpy as np
import pandas as pd
import argparse


def argparser():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--original_video_info_csv", required=True,
        help="path to data_.csv Header must include uid, subject_id, video_path.")
    ap.add_argument("--segm_video_info_dir", required=True,
        help="path to data_.csv Header must include uid, subject_id, video_path.")    
    ap.add_argument("--output_csv", required=True,
        help="path to data_.csv Header must include uid, subject_id, video_path.")
    ap.add_argument("--vid_path_prefix", required=False,
        default="",
        help="prefix to append to video path.")
    args = ap.parse_args()
    return args

def read_transcription_to_df(path):
    full_utt_id = []
    text = []
    with open(path, 'r') as f:
        for line in f:
            full_utt_id.append(line.split(" ")[0])
            text.append(" ".join(line.split(" ")[1:]).split("\n")[0])

    df = pd.DataFrame({'full_utt_id': np.array(full_utt_id), 'full_utt_text': np.array(text)})
    return df


def main():
    args = argparser()
    videos_info = pd.read_csv(args.original_video_info_csv)
    videos_info.columns=["keyword", "videoID", "channelID","diagnosis","gender", "age", "role", "category", "speakerID", "ti", "tf", "totals" ]
    # header: keyword, videoID, channelID, diagnosis, gender, age, role, category, speakerID, ti, tf, totals

    mp4_df = pd.DataFrame(columns=['uid', 'video_path'])
    utt2spk_df = pd.DataFrame(columns=['uid', 'speakerID'])
    text_df = pd.DataFrame(columns=['full_utt_id', 'full_utt_text'])
    
    for subj in videos_info.speakerID.values:
        mp4_path = args.segm_video_info_dir + '/' + subj + "/mp4.scp"
        utt2spk_path = args.segm_video_info_dir + '/' + subj + "/utt2spk"
        text_path = args.segm_video_info_dir + '/' + subj + "/text"
        
        try:
            mp4 = pd.read_csv(mp4_path, header=None, sep=' ', names=['uid', 'video_path'])
        except:
            with open('logs/make_data_csv_log.log', 'a') as f:
                f.write("Could not find mp4 info files for subject " + subj + ". Skipping it. \n")
            continue

        try:
            utt2spk = pd.read_csv(utt2spk_path, header=None, sep=' ', names=['uid', 'speakerID'])
        except:
            with open('logs/make_data_csv_log.log', 'a') as f:
                f.write("Could not find utt2spk info files for subject " + subj + ". Skipping it. \n")
            continue

        try:
            text = read_transcription_to_df(text_path)
        except:
            with open('logs/make_data_csv_log.log', 'a') as f:
                f.write("Could not find text info files for subject " + subj + ". Skipping it. \n")
            continue


        mp4_df = mp4_df.append(mp4)
        utt2spk_df = utt2spk_df.append(utt2spk)
        text_df = text_df.append(text)

    utt2spk_df['full_utt_id'] = utt2spk_df['uid'].str.rsplit('-', n=1, expand=True)[0]
    mp4_df['video_path'] = args.vid_path_prefix + mp4_df['video_path']

    # merge all dfs:
    df = pd.merge(mp4_df, utt2spk_df, on=['uid'])
    df = pd.merge(df, text_df, on=['full_utt_id'])
    df = pd.merge(df, videos_info, on=['speakerID'])

    # make columns in pre-defined order
    df_final = df[['uid', 'speakerID', 'videoID', 'channelID', 'full_utt_id', 'full_utt_text', 'video_path', 'keyword', 'diagnosis', 'gender', 'age',"ti", "tf"]]
    
    # save final df
    df_final.to_csv(args.output_csv, index=False)






if __name__ == "__main__":
        main()    
