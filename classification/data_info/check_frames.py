#Margarida Ferro, April 2023
import pandas as pd
import numpy as np


data_csv = 'final_data.csv'

df = pd.read_csv(data_csv)
frames = df.num_frames.values

good_frames = []

for frame in frames:
    if frame >= 24:
        good_frames.append("true")
    else:
        good_frames.append("false")

print(len(good_frames))

df['valid'] = good_frames
df.to_csv('final_data.csv', index=False) 


