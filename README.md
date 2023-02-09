# Music Classifier project
## Introduction 
This project is a Kaggle competition available here : https://www.kaggle.com/competitions/tsma-202223-music-genre-classification/data
We have a few music extracts to classify in genres, we can't see them in advance. Each genre is represented by a number as follows : 
1. Classical
2. Electronic
3. Folk
4. Hip-Hop
5. WorldMusic
6. Experimental
7. Pop
8. Rock
We have also a few extracts called "train extracts" and for each we know their genre. From these extracts we will create different classifier and try to find the best one. The result is organised in a csv file with two columns, each song is represented by 6 consecutive numbers (the first column), associated to his genre (a number from 1 to 8). The project is developed in Python. 

## One of the most basic classifier : kNN-classifier with MFCC
Here we are calculating for each extract what we call MFCC which is a serie of coefficients used to describe a song. So we just compute a distance between each extract to classify and all train extracts. After that we keep only k closest distances (the best k we found is 5), and look for the genre which is the most present.
We have an accuracy around 34 %. 

## Neural network classifier with a vector of 512 values
This method uses in enter a vector of 512 values to characterize an extract, obtained with librosa in Python.
So here we just create a deep neural network with dense layers. The code available is the final version, and we obtained around 46 % accuracy. 

## Decision tree classifier with same vectors as before
We use here a decsion tre which is working like a neural network, because we have to choose number of epoch of training, and some hyper parameters. 
We obtained an accuracy around 61% which is clearly better. 

## CNN and transfer learning 
In input, we have sounds and we are using pre-trained network like VGGish or OpenL3. These networks are creating spectrograms which are considered as pictures and can pass trough convolutionnal neural network. After that we created our own neural network and try to optimize it. The better accuracy we obtained is around 67%. 
Of course, we tried data augmentation to have a better accuracy or ensemble learning but the last one didn't work well. 

The csv file linked contains the ebst results. 
