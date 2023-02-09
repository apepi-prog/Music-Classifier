import os 

for file in os.listdir('/kaggle/working'):
    if (not file.endswith('.virtual_documents')):
        os.remove(file)

# Melspectro Hop_size 2048, shape 313x128
!wget http://dept-info.labri.fr/~hanna/ProjetClassif/melspectro_songs_train_new.pickle
!wget http://dept-info.labri.fr/~hanna/ProjetClassif/melspectro_genres_train_new.pickle

!wget http://dept-info.labri.fr/~hanna/ProjetClassif/melspectro_songs_test_new.pickle
!wget http://dept-info.labri.fr/~hanna/ProjetClassif/melspectro_filenames_test.pickle

import pickle
from sklearn.model_selection import train_test_split
import tensorflow_io as tfio
import numpy as np

x_songs = pickle.load(open('melspectro_songs_train_new.pickle','rb'))
y_genres = pickle.load(open('melspectro_genres_train_new.pickle','rb'))

#Data Augmentation 
print("Data Augmentation starting :")
freqs_masks = []
times_masks = []
print("Creating new data ...")
for song in x_songs:
    # Freq masking
    freq_m = tfio.audio.freq_mask(song, param=50)
    freqs_masks.append(freq_m)
    
    # Time masking
    time_m = tfio.audio.time_mask(song, param=10)
    times_masks.append(time_m)

print("Adding to x_songs and y_genres ...")
x_songs = np.append(x_songs, freqs_masks, axis=0)
y_g = y_genres
y_genres = np.append(y_genres, y_genres, axis=0)
x_songs = np.append(x_songs, times_masks, axis=0)
y_genres = np.append(y_genres, y_g, axis=0)
print("Data augmentation finished")
del freqs_masks
del times_masks

#splitting our data (25% used for testing)
x_train, x_test, y_train, y_test = train_test_split(x_songs, y_genres, test_size=0.25)

#reshape needed for conv layers
input_size_x = x_songs.shape[1]
input_size_y = x_songs.shape[2]
del x_songs 
del y_genres

x_train = x_train.reshape(-1, input_size_x, input_size_y, 1)
x_test = x_test.reshape(-1, input_size_x, input_size_y, 1)

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, BatchNormalization, Dropout, Flatten, Conv2D, MaxPooling2D

batch_size = 32
num_classes = 8
epochs = 20

#Define model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

#compile model 
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# fit the model 
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)

#evaluate accuracy
_, accuracy = model.evaluate(x_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))

#Writing in csv file
import csv
import pickle

to_pred_songs = pickle.load(open('melspectro_songs_test_new.pickle','rb'))
to_pred_songs = to_pred_songs.reshape(-1, to_pred_songs.shape[1], to_pred_songs.shape[2], 1)
to_pred_track_names = pickle.load(open('melspectro_filenames_test.pickle','rb'))

#Creating our result csv
with open('submisission_cnn.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    
    # write the header
    header = ['track_id', 'genre_id']
    writer.writerow(header) 
    
    predictions = model.predict(to_pred_songs)
    for i in range(len(to_pred_track_names)):
        id_genre = np.argmax(predictions[i]) + 1
        d = [to_pred_track_names[i], id_genre]
        ## write predictions here
        # data like this id_song, id_genre
        writer.writerow(d)
    writer.writerow(['59684', 1])
    writer.writerow(['98565', 1])
    writer.writerow(['98568', 1])
    writer.writerow(['98569', 1])
    writer.writerow(['98571', 1])
    writer.writerow(['98559', 1])
    
print("Predictions written in submisission_cnn.csv")

## Transfert Learning 
from tensorflow.keras.applications.resnet50 import ResNet50
import pickle
import csv
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, BatchNormalization, Dropout, Flatten, Conv2D, MaxPooling2D

to_pred_songs = pickle.load(open('melspectro_songs_test_new.pickle','rb'))
to_pred_songs = to_pred_songs.reshape(-1, to_pred_songs.shape[1], to_pred_songs.shape[2], 1)
to_pred_songs = tf.image.grayscale_to_rgb(tf.convert_to_tensor(to_pred_songs))
to_pred_track_names = pickle.load(open('melspectro_filenames_test.pickle','rb'))

resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(313,128,3))
model = Sequential()
model.add(resnet)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(8, activation='softmax'))
model.summary()

#compile model 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(x_train.shape)
x_train = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_train))
x_test = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_test))
# fit the model 
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=32)

#evaluate accuracy
_, accuracy = model.evaluate(x_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))

with open('submisission_cnn.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    
    # write the header
    header = ['track_id', 'genre_id']
    writer.writerow(header) 
    
    predictions = model.predict(to_pred_songs)
    print(predictions.shape)
    for i in range(len(to_pred_track_names)):
        id_genre = np.argmax(predictions[i]) + 1
        d = [to_pred_track_names[i], id_genre]
        ## write predictions here
        # data like this id_song, id_genre
        writer.writerow(d)
    writer.writerow(['59684', 1])
    writer.writerow(['98565', 1])
    writer.writerow(['98568', 1])
    writer.writerow(['98569', 1])
    writer.writerow(['98571', 1])
    writer.writerow(['98559', 1])
    
print("Predictions written in submisission_cnn.csv")