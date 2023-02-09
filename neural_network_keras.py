# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, BatchNormalization, Dropout

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#!wget http://dept-info.labri.fr/~hanna/ProjetClassif/features_adapte.csv
#!wget http://dept-info.labri.fr/~hanna/ProjetClassif/features_head.csv
#!wget http://dept-info.labri.fr/~hanna/ProjetClassif/train_clean.csv

head_features = '/kaggle/working/features_head.csv'
train_clean = '/kaggle/working/train_clean.csv'
test_clean = '/kaggle/input/tsma-202223-music-genre-classification/test.csv'
adapt_features = '/kaggle/working/features_adapte.csv'

# Nom des features
features = pd.read_csv(filepath_or_buffer=head_features, sep=",")

# Jointure features/tracks du dataset train
traingenre = pd.read_csv(filepath_or_buffer=train_clean, sep=",")
iter_csv = pd.read_csv(filepath_or_buffer=adapt_features, sep=",", iterator=True, chunksize=10000)
datatrain = pd.concat([chunk for chunk in iter_csv])

#Jointure des features/tracks du datastet test
test = pd.read_csv(filepath_or_buffer=test_clean, sep=",")
print(len(test))
iter_csv = pd.read_csv(filepath_or_buffer=adapt_features, sep=",", iterator=True, chunksize=10000)
datatest = pd.concat([chunk for chunk in iter_csv])

data = pd.merge(traingenre, datatrain, on='track_id')

#each extract got his descriptors values (519)
x = data.drop('genre_id',axis=1).values

#genre id associated to extract (between 1 and 8)
Y = data['genre_id'].values

x_train, x_test, y_train, y_test = train_test_split(x, Y)

## files to predict 
to_predict = pd.merge(test, datatest, on='track_id').values
print(to_predict.shape)

batch_size = 32
num_classes = 9
nb_features = 519
epochs = 300

#Define model
model = Sequential()
model.add(Dense(1024, input_shape=(nb_features, ), activation='relu'))
model.add(BatchNormalization())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(16, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dense(num_classes, activation='softmax'))

#compile model 
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the model 
model.fit(x, Y, epochs=epochs, batch_size=batch_size)

#evaluate accuracy
_, accuracy = model.evaluate(x_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))

#Creating our result csv
with open('submisission_nn_519features.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    
    # write the header
    header = ['track_id', 'genre_id']
    writer.writerow(header) 

    id_songs = [str(test.iloc[i][0]).rjust(6, '0') for i in range(len(test))]
    predictions = model.predict(to_predict)
    for i in range(len(test)):
        id_genre = np.argmax(predictions[i])
        d = [id_songs[i], id_genre]
        ## write predictions here
        # data like this id_song, id_genre
        writer.writerow(d)
        
print("Predictions written in submisission_nn_519features.csv")