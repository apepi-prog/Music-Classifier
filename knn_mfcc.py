import numpy as np 
import pandas as pd 
import sys
import os
from pydub import AudioSegment
import torch
import torchaudio
from torchaudio import transforms
import csv
import collections

## remove warnings from loading files
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
directory_train = '/kaggle/input/tsma-202223-music-genre-classification/train/Train/'
directory_test = '/kaggle/input/tsma-202223-music-genre-classification/test/Test/'
csv_genres = '/kaggle/input/tsma-202223-music-genre-classification/genres.csv'
csv_train_genres = '/kaggle/input/tsma-202223-music-genre-classification/train.csv'

for file in os.listdir('/kaggle/working'):
    if (file.endswith('.csv')):
        os.remove(file)

#Creating our result csv
with open('submisission_knn_mfccs.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    
    # write the header
    header = ['track_id', 'genre_id']
    writer.writerow(header)

## Initializing our genre dictionnary with id as key and genre as value
df = pd.read_csv(csv_genres)
genres = {}
for i in range(len(df)):
    genres[df.iloc[i][0]] = df.iloc[i][1]
    
    
## Reading type for each music and stocking it in dictionnary
train_genres = {}
df = pd.read_csv(csv_train_genres)
for i in range(len(df)):
    train_genres[str(df.iloc[i][0]).rjust(6, '0')] = df.iloc[i][1]

## Dictionnary for mfccs on train sounds
mfccs_train = {}
for f in os.listdir(directory_train):
    if f.endswith(".mp3"):
            name = directory_train + f
        #conversion in wav and adding mfcc if it works
        try:
            sound = AudioSegment.from_file(name, format="mp3")
            dst = f[0:6] + '.wav'
            sound.export(dst, format="wav")
            # getting sound and sample rate  
            waveform, sample_rate = torchaudio.load(dst, normalize=True)
            os.remove(dst)
            transform = transforms.MFCC(sample_rate=sample_rate, n_mfcc=20,)
            # mfcc shape is like (x, y) 
            # where y is the number of time frames, given by the length of the audio (in samples) divided by the hop_length
            # x is the number of mfcc coefficients
            mfcc_mat = transform(waveform)
            mfcc = []
            for i in range(mfcc_mat.size()[1]):
                if (mfcc_mat.size()[0] == 2):
                    mfcc_mat[0] = (mfcc_mat[0][i] + mfcc_mat[1][i]) / 2
                mfcc.append((torch.sum(mfcc_mat[0][i])/mfcc_mat.size()[2]).item())
            mfccs_train[f[0:6]] = mfcc
        except:
            print("Error to convert this train file", f)
        
print("Mfcc <train> computed")

## Dictionnary for mfccs on test sounds
mfccs_test = {}
for f in os.listdir(directory_test):
    if f.endswith(".mp3"):
        try:
            sound = AudioSegment.from_file(directory_test+f, format="mp3")
            dst = f[0:6] + '.wav'
            sound.export(dst, format="wav")
            # getting sound and sample rate  
            waveform, sample_rate = torchaudio.load(dst, normalize=True)
            os.remove(dst)
            transform = transforms.MFCC(sample_rate=sample_rate, n_mfcc=20,)
            # mfcc shape is like (x, y) 
            # where y is the number of time frames, given by the length of the audio (in samples) divided by the hop_length
            # x is the number of mfcc coefficients
            mfcc_mat = transform(waveform)
            mfcc = []
            for i in range(mfcc_mat.size()[1]):
                if (mfcc_mat.size()[0] == 2):
                    mfcc_mat[0] = (mfcc_mat[0][i] + mfcc_mat[1][i]) / 2
                mfcc.append((torch.sum(mfcc_mat[0][i])/mfcc_mat.size()[2]).item())
            mfccs_test[f[0:6]] = mfcc
        except:
            mfcc = []
            for i in range(20):
                mfcc.append(0.0)
            mfccs_test[f[0:6]] = mfcc
            print("Error to convert this test file", f, "<added to mfccs but with 0.0>")

print("Mfcc <test> computed")

#can be changed but one of the best value is 10
k_neighbours = 5

mfccs_test_sorted = collections.OrderedDict(sorted(mfccs_test.items()))
for k_test, value in mfccs_test_sorted.items():
    dist_arr = []
    style_dist = {}
    for k, v in mfccs_train.items():
        value = np.array(value)
        v = np.array(v)
        dist = np.sqrt(np.sum(np.square(value - v)))
        dist_arr.append(dist)
        style_dist[dist] = train_genres[k]
    dist_arr.sort()
    counts = {}
    for key in genres.keys():
        counts[key] = 0
    for j in range(k_neighbours):
        counts[style_dist[dist_arr[j]]] += 1
    genres_numbers = list(counts.values())
    genres_k = list(counts.keys())
    max_i = genres_k[genres_numbers.index(max(genres_numbers))]
    del dist_arr
    with open('submisission_knn_mfccs.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
    
        # data like this id_song, id_genre
        data = [k_test, max_i]
        writer.writerow(data)

print("Predictions written in submisission_knn_mfccs.csv")
    