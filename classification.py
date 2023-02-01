import numpy as np
import sys, glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import itertools
from math import *
import pandas as pd
import ast
import os
import cv2
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image

train_df = pd.read_csv(r'D:\Cours\master\semestre 3\Machine_learning\Advanced-machine-learning-project\Data\train_df.csv')
train_df = train_df.dropna(subset=["R", "G", "B"])

indexes_to_drop = []
for index, row in train_df.iterrows():
    R = [int(x) for x in row["R"].strip('[]').split() if x != '...']
    G = [int(x) for x in row["G"].strip('[]').split() if x != '...']
    B = [int(x) for x in row["B"].strip('[]').split() if x != '...']
    if len(R) != len(G) or len(R) != len(B) or len(G) != len(B):
        indexes_to_drop.append(index)
    else:
        combined = list(zip(R, G, B))
        train_df.at[index, "pixels"] = combined

train_df.drop(index=indexes_to_drop, inplace=True)
train_df = train_df.reset_index(drop=True)

print(train_df.to_string())
