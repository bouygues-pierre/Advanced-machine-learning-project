import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
from PIL import Image
import numpy as np

project_root = '/Volumes/Pierre_2TO/Cours/master/semestre 3/Machine_learning/Advanced-machine-learning-project'
data_path = r'D:\Cours\master\semestre 3\Machine_learning\Advanced-machine-learning-project\Data'
train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
sample_df = pd.read_csv(os.path.join(data_path, 'sample_submission.csv'))

train_df['image'] = train_df['Image_Label'].apply(lambda x: x.split('_')[0])
train_df['label'] = train_df['Image_Label'].apply(lambda x: x.split('_')[1])
train_df = train_df.drop(['Image_Label'], axis=1)  # split the image name and label into two different column
train_df = train_df.dropna(subset=['EncodedPixels'])  # delete all the NaN masks
train_df = train_df.drop(["EncodedPixels"], axis=1)  # delete the column encoded pixel we will use thos data here
train_df = train_df.drop_duplicates(subset='image', keep=False)  # delete image with more than one mask to simplify
# the problem
train_df['Flower_or_not_Flower'] = train_df['label'].apply(lambda x: 1 if x == 'Flower' else 0)  # add a column with
# 1 if flower and 0 if not flower
train_df = train_df.reset_index(drop=True)  # reset index

# #create new columns in the dataframe to store R, G, and B values
train_df['pixels'] = None

#
# plt.figure()
# image_path = r'D:\Cours\master\semestre 3\Machine_learning\Advanced-machine-learning-project\Data\train_images\044b59c.jpg'
# image = cv2.imread(image_path)
# plt.imshow(image)
# plt.show()

#iterate through the dataframe
for index, row in train_df.iterrows():
    print(index)
    #  read in the image using opencv
    image_name = row['image']
    image_path = fr'D:\Cours\master\semestre 3\Machine_learning\Advanced-machine-learning-project\Data\train_images\{image_name} '
    print(image_path)
    if os.path.isfile(image_path):
        image = cv2.imread(image_path)
        print('image read')
        R, G, B = cv2.split(image)
    else:
        print(f"image not found : {image_name}")
    print('image read')
    R, G, B = cv2.split(image)
    #  get the pixel values of the image
    #  add the pixel values to the dataframe
    # train_df.at[index, 'pixels'] = list(zip(R, G, B))

    print('image added')

train_df.to_csv(r"D:\Cours\master\semestre 3\Machine_learning\Advanced-machine-learning-project\Data\train_df_3.csv", index=False)

# # Read the image data into a numpy array
# pixel_value = []
# for img_name in train_df['image']:
#     img_path = fr'D:\Cours\master\semestre 3\Machine_learning\Advanced-machine-learning-project\Data\train_images\{img_name}'
#     img = Image.open(img_path)
#     print(f'image {img_name} append')
#     pixel_value.append(np.array(img))
# print('off')
# pixel_value = np.array(pixel_value)
# print('saving...')
# np.save('Pixel_value.npy', pixel_value)
