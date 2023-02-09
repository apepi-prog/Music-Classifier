# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import csv
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

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
#print(features.columns)
#print(features)

# Jointure features/tracks du dataset train
traingenre = pd.read_csv(filepath_or_buffer=train_clean, sep=",")
iter_csv = pd.read_csv(filepath_or_buffer=adapt_features, sep=",", iterator=True, chunksize=10000)
datatrain = pd.concat([chunk for chunk in iter_csv])

#Jointure des features/tracks du datastet test
test = pd.read_csv(filepath_or_buffer=test_clean, sep=",")

iter_csv = pd.read_csv(filepath_or_buffer=adapt_features, sep=",", iterator=True, chunksize=10000)
datatest = pd.concat([chunk for chunk in iter_csv])

data = pd.merge(traingenre, datatrain, on='track_id')
#print(data.shape, traingenre.shape, datatrain.shape)

# exemple
#TRACK = 115
##print(datatrain.values[TRACK])

#training
#each extract got his descriptors values (519)
x = data.drop('genre_id',axis=1).values
#print(x)

#print(xtrain)
#genre id associated to extract (between 1 and 8)
Y = data['genre_id'].values

x_tr, x_t, y_tr, y_t = train_test_split(x, Y)

x_train = [(a,b) for a,b in zip(x_tr, y_tr)]
x_test = [(a,b) for a,b in zip(x_t, y_t)]

## files to predict 
to_predict = pd.merge(test, datatest, on='track_id')
x_to_pred = data.drop('genre_id',axis=1).values
x_predict = [torch.tensor(a) for a in x_to_pred]

batch_size = 32
train_dataloader = DataLoader(x_train, batch_size=batch_size, shuffle=True, num_workers=2)
test_dataloader = DataLoader(x_test, batch_size=batch_size, num_workers=2)

num_classes = 9
nb_features = 519
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.eval()
        self.l1 = nn.Linear(nb_features, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.Linear(64, num_classes)
        self.drop = nn.Dropout(0.2)
        self.sig = nn.Sigmoid()
        ## TODO test other activation function like Softmax, Tanh 
        
    def forward(self , x):
        x = self.l1(x)
        x = self.sig(x)
        x = self.l2(self.drop(x))
        x = self.sig(x)
        x = self.l3(self.drop(x))
        x = self.sig(x)
        x = self.sig(self.l4(self.drop(x)))

        return x

model = NeuralNetwork().float()
print(model)

lr = 0.00001
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr)

## Train
epochs = 9

#arrays to get our plots at the end
x_points = torch.arange(epochs)
loss_points = torch.zeros(epochs)
acc_points = torch.zeros(epochs)

for e in range(epochs):
    size = len(train_dataloader.dataset)
    num_batches = len(train_dataloader)
    
    running_acc = 0
    running_loss = 0
    total = 0
    for batch, (X, y) in enumerate(train_dataloader):
        
        ## one-hot encoding our y data
        y = nn.functional.one_hot(y, num_classes).to(torch.float32)

        #forward pass & loss
        pred = model(X.float())
        loss = loss_fn(pred, y)
        total += y.size(0)

        #backward pass / backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        #computing accuracy
        y_hat = torch.zeros(len(y), num_classes)
        for i in range(len(y)):
            ind_max = torch.argmax(pred[i])
            y_hat[i][ind_max] = 1
            
        
        for i in range(len(y)):
            for j in range(num_classes):
                if (y[i][j] == y_hat[i][j]):
                    running_acc += 1              

    loss = running_loss / len(train_dataloader)
    accuracy = 100 * running_acc / (total*num_classes)

    loss_points[e] = loss
    acc_points[e] = accuracy

    print(f'{e}/{epochs}: loss={loss:.4f} acc={accuracy:.4f}')
print("\n")

running_loss = 0
running_acc = 0
total = 0
print("Evaluating model ...")
with torch.no_grad():
    model.eval()
    for batch, (X, y) in enumerate(test_dataloader):

        ## one-hot encoding our y data
        y = nn.functional.one_hot(y, num_classes).to(torch.float32)    

        #forward pass & loss
        pred = model(X.float())
        total += y.size(0)
        running_loss += loss_fn(pred, y).item()

        #computing accuracy
        y_hat = torch.zeros(len(pred), num_classes)
        for i in range(len(y)):
            ind_max = torch.argmax(pred[i])
            y_hat[i][ind_max] = 1
        
        for i in range(len(y)):
            for j in range(num_classes):
                if (y[i][j] == y_hat[i][j]):
                    running_acc += 1 

print("Loss during test :", running_loss/total )
print("Accuracy of the model on test set : ", (100 * running_acc) / (total*num_classes), "%\n")

#Creating our result csv
with open('submisission_nn_519features.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    
    # write the header
    header = ['track_id', 'genre_id']
    writer.writerow(header) 

    id_songs = [str(test.iloc[i][0]).rjust(6, '0') for i in range(len(test))]
    for i in range(len(x_predict)):
        x_predict[i] = x_predict[i].float()
        prediction = model(x_predict[i])
        id_genre = torch.argmax(prediction)
        d = [id_songs[i], id_genre.item()]
        ## write predictions here
        # data like this id_song, id_genre
        writer.writerow(d)
        
print("Predictions written in submisission_nn_519features.csv")
    
