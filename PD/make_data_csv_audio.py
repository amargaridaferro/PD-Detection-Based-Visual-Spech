# general packages
import numpy as np
import pandas as pd
import argparse
import os

def argparser():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--original_video_info_csv", required=True,
        help="path to data_.csv Header must include uid, subject_id, video_path.")
    ap.add_argument("--segms_dir", required=True,
        help="path to data_.csv Header must include uid, subject_id, video_path.")    
    ap.add_argument("--output_csv", required=True,
        help="path to data_.csv Header must include uid, subject_id, video_path.")
    args = ap.parse_args()
    return args


def main():
    args = argparser()
    videos_info = pd.read_csv(args.original_video_info_csv)
    videos_info.columns=["yt_vid", "yt_chid", "wsm_keyword","subject_id", "diagnosis", "gender", "age", "included"]
    # header: yt_id,channel,wsm_keyword,speaker_id,diagnosis,gender,age,included

    data_df = pd.DataFrame(columns=['uid', 'subject_id', 'full_utt_id', 'wav_path']) 
    
    for subj in videos_info.subject_id.values: 
        subj_dir=args.segms_dir + '/' + subj + '/'
        files = os.listdir(subj_dir) # [''c_002-00026-00004.mp4', ...]

        paths = np.array([ subj_dir + f for f in files])
        uids = np.array([f.split('.')[0] for f in files])
        full_utt_ids = np.array(['_'.join(u.split('_')[:-1]) for u in uids])
        subject_ids = np.array([subj for i in range(len(uids))])
        
        df = pd.DataFrame({'uid': uids, 'subject_id': subject_ids, 'full_utt_id': full_utt_ids, 'wav_path': paths})
        
        data_df = data_df.append(df)

    # merge all dfs:
    df = pd.merge(data_df, videos_info, on=['subject_id'])
    # make columns in pre-defined order
    df_final = df[['uid', 'subject_id', 'yt_vid', 'yt_chid', 'full_utt_id', 'wav_path', 'wsm_keyword', 'diagnosis', 'gender', 'age', 'included']]
    
    # save final df
    df_final.to_csv(args.output_csv, index=False)






if __name__ == "__main__":
        main()    
