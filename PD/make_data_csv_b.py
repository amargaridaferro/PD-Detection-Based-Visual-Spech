# general packages
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
    args = ap.parse_args()
    return args


def main():
    args = argparser()
    videos_info = pd.read_csv(args.original_video_info_csv)
    videos_info.columns=["yt_vid", "yt_chid", "wsm_keyword","subject_id", "diagnosis", "gender", "age"]
    # header: yt_id,channel,wsm_keyword,speaker_id,diagnosis,gender,age

    data_df = pd.DataFrame(columns=['uid', 'subject_id', 'full_utt_id', 'full_utt_text', 'video_path']) 
    
    for subj in videos_info.subject_id.values: 
        data_info_path = args.segm_video_info_dir + '/' + subj + "_data.csv"
        
        try:
            data_info = pd.read_csv(data_info_path)
        except:
            with open('logs/make_data_csv_b_log.log', 'a') as f:
                f.write("Could not find data info files for subject " + subj + ". Skipping it. \n")
            continue
        
        data_df = data_df.append(data_info)

    # merge all dfs:
    df = pd.merge(data_df, videos_info, on=['subject_id'])
    # make columns in pre-defined order
    df_final = df[['uid', 'subject_id', 'yt_vid', 'yt_chid', 'full_utt_id', 'full_utt_text', 'video_path', 'wsm_keyword', 'diagnosis', 'gender', 'age']]
    
    # save final df
    df_final.to_csv(args.output_csv, index=False)






if __name__ == "__main__":
        main()    
