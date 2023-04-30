import os
import sys
import numpy as np
import pandas as pd


def main(args):
    subj = args[0]
    mp4_dir = "/cfs/tmp/mctb/projects/osa_multimodal//data/segmented_videos/vad/" + subj + "/"
    out_cvs = "/cfs/tmp/mctb/projects/osa_multimodal/data_info/WOSA/segmented_videos/vad/" +  subj + "_data.csv"

    files = os.listdir(mp4_dir) # [''c_002-00026-00004.mp4', ...]

    paths = np.array([mp4_dir + f for f in files])
    uids = np.array([f.split('.')[0] for f in files])
    full_utt_ids = np.array(['-'.join(u.split('-')[:-1]) for u in uids])
    subject_ids = np.array([subj for i in range(len(uids))])
    texts = np.array(["NI" for i in range(len(uids))])
    
    df = pd.DataFrame({'uid': uids, 'subject_id': subject_ids, 'full_utt_id': full_utt_ids, 'full_utt_text': texts, 'video_path': paths})
    df.to_csv(out_cvs, index=False)



if __name__ == '__main__':
    main(sys.argv[1:])
