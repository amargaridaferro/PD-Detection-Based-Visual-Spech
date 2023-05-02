#Based on Catarina Botelho work
#Margarida Ferro, April 2023
import pandas as pd
import numpy as np


data_csv = 'data_only_faces.csv'
output_data_csv ='final_data.csv'

df = pd.read_csv(data_csv)
paths = df.lipread_emb_path.values

print(paths)

l = []
discard = []
n_iter = len(paths)


l = []
discard = []
n_iter = len(paths)
for i, p in enumerate(paths):
    print ('[', i+1, '/', n_iter, ']', end='\r')
    try:
        l.append(len(np.load(p)['data'].squeeze()))
    except FileNotFoundError:
        print ()
        print ('File ', p, 'not found. Removing it ffrom data.csv')
        discard.append(p)

print (len(discard), ' files were not found.')

df = df[~df.lipread_emb_path.isin(discard)]
df['num_frames'] = l
df.to_csv(output_data_csv, index=False) 