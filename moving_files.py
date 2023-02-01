import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
from PIL import Image
import numpy as np
import os

df = pd.read_csv('D:/Cours/master/semestre 3/Machine_learning/Advanced-machine-learning-project/Data/train_df.csv')
flower_dir = 'D:/Cours/master/semestre 3/Machine_learning/Advanced-machine-learning-project/Data/train_image_flower'
not_flower_dir = 'D:/Cours/master/semestre 3/Machine_learning/Advanced-machine-learning-project/Data/train_image_notflower'
data_path = 'D:/Cours/master/semestre 3/Machine_learning/Advanced-machine-learning-project/Data/train_images'

# Iterate through the rows of the dataframe
for index, row in df.iterrows():
    # Get the image path and label
    label = row['label']
    image_path = os.path.join(data_path, row['image'])

    print(index, row['image'])

    # Move the image to the appropriate directory
    if label == 'Flower':
        os.rename(image_path, os.path.join(flower_dir, row['image']))
    elif label in ['Sugar', 'Gravel', 'Fish']:
        os.rename(image_path, os.path.join(not_flower_dir, row['image']))

    print('image {} mooved'.format(row['image']))
