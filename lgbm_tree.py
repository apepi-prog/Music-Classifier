# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
from sklearn.model_selection import train_test_split
import lightgbm as lgb

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
iter_csv = pd.read_csv(filepath_or_buffer=adapt_features, sep=",", iterator=True, chunksize=10000)
datatest = pd.concat([chunk for chunk in iter_csv])

data = pd.merge(traingenre, datatrain, on='track_id')

#data use to our model training and test
#each extract got his descriptors values (519)
x = data.drop('genre_id',axis=1).values

#genre id associated to extract (between 1 and 8)
Y = data['genre_id'].values

## files to predict 
to_predict = pd.merge(test, datatest, on='track_id').values

x_train, x_test, y_train, y_test = train_test_split(x, Y, test_size=0.10)

# model LGBM tree
model = lgb.LGBMClassifier(learning_rate=0.11, max_depth=50, num_leaves=100, num_iterations=50)
model.fit(x_train, y_train, eval_set=[(x_test,y_test),(x_train,y_train)], eval_metric='logloss')

print('Training accuracy {:.4f}'.format(model.score(x_train,y_train)))
print('Testing accuracy {:.4f}'.format(model.score(x_test,y_test)))

#Creating our result csv
with open('submisission_lgbm.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    
    # write the header
    header = ['track_id', 'genre_id']
    writer.writerow(header) 

    id_songs = [str(test.iloc[i][0]).rjust(6, '0') for i in range(len(test))]
    predictions = model.predict(to_predict)
    print(predictions.shape)
    for i in range(len(test)):
        id_genre = predictions[i]
        d = [id_songs[i], id_genre]
        ## write predictions here
        # data like this id_song, id_genre
        writer.writerow(d)
        
print("Predictions written in submisission_lgbm.csv")

#With 0.09 lr we got 62 % accuracy here (not on the predictions to do for real)
#0.1 lr, depth=6 and num_leaves=17 num_iter = 80 we got 60 % accuracy on the real tests