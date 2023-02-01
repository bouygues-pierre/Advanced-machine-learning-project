import numpy as np  # linear algebra
import pandas as pd

pd.set_option("display.max_rows", 100)
import os

import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 14
import cv2

project_root = '/Volumes/Pierre_2TO/Cours/master/semestre 3/Machine_learning/Advanced-machine-learning-project'
data_path = '/Volumes/Pierre_2TO/Cours/master/semestre 3/Machine_learning/Data'
train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
sample_df = pd.read_csv(os.path.join(data_path, 'sample_submission.csv'))

# ----------------------------------------------------------------------------------------------------------------------

#                                               Data visualisation

# ----------------------------------------------------------------------------------------------------------------------

print(train_df.head())

print(f'There are {train_df.shape[0]} records in train.csv')

plt.figure()
train_df['Image_Label'].apply(lambda x: x.split('_')[1]).value_counts().plot(kind='bar')

len_train = len(os.listdir(os.path.join(data_path, "train_images")))
len_test = len(os.listdir(os.path.join(data_path, "test_images")))
print(f'There are {len_train} images in train dataset')
print(f'There are {len_test} images in test dataset')

# Figuring out the total number of images having empty masks.
print(f'There are {len(train_df[train_df["EncodedPixels"].isnull()])} images without mask')

plt.figure()
train_df.loc[train_df['EncodedPixels'].isnull(), 'Image_Label'].apply(lambda x: x.split('_')[1]).value_counts().plot(
    kind="bar")

train_df.loc[train_df['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[1]).value_counts()

plt.figure()
train_df.loc[train_df['EncodedPixels'].isnull() == False, 'Image_Label'].apply(
    lambda x: x.split('_')[0]).value_counts().value_counts().plot(kind="bar")

palet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249, 50, 12)]


def name_and_mask(start_idx):
    col = start_idx
    img_names = [str(i).split("_")[0] for i in train_df.iloc[col:col + 4, 0].values]
    if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):
        raise ValueError

    labels = train_df.iloc[col:col + 4, 1]
    mask = np.zeros((1400, 2100, 4), dtype=np.uint8)

    for idx, label in enumerate(labels.values):
        if label is not np.nan:
            mask_label = np.zeros(2100 * 1400, dtype=np.uint8)
            label = label.split(" ")
            positions = map(int, label[::2])
            length = map(int, label[1::2])
            for pos, le in zip(positions, length):
                mask_label[pos:(pos + le)] = 1
            mask[:, :, idx] = mask_label.reshape(1400, 2100)
    return img_names[0], mask


def show_mask_image(col):
    name, mask = name_and_mask(col)
    img = cv2.imread(str(train_df / name))
    fig, ax = plt.subplots(figsize=(15, 15))

    for ch in range(4):
        contours, _ = cv2.findContours(mask[:, :, ch], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for i in range(len(contours)):
            cv2.polylines(img, contours[i], True, palet[ch], 2)
    ax.set_title(name)
    ax.imshow(img)
    plt.show()


idx_no_class = []
idx_class_1 = []
idx_class_2 = []
idx_class_3 = []
idx_class_4 = []
idx_class_multi = []
idx_class_triple = []
idx_no_defect = []

for col in range(0, len(train_df), 4):
    img_names = [str(i).split("_")[0] for i in train_df.iloc[col:col + 4, 0].values]
    if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):
        raise ValueError

    labels = train_df.iloc[col:col + 4, 1]
    if labels.isna().all():
        idx_no_defect.append(col)
    elif (labels.isna() == [False, True, True, True]).all():
        idx_class_1.append(col)
    elif (labels.isna() == [True, False, True, True]).all():
        idx_class_2.append(col)
    elif (labels.isna() == [True, True, False, True]).all():
        idx_class_3.append(col)
    elif (labels.isna() == [True, True, True, False]).all():
        idx_class_4.append(col)
    elif labels.isna().sum() == 1:
        idx_class_triple.append(col)
    else:
        idx_class_multi.append(col)

for idx in idx_class_2[:5]:
    print(type(idx))
    show_mask_image(idx)
