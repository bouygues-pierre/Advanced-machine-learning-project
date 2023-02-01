import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import cv2


# ----------------------------------------------------------------------------------------------------------------------

#                                                      Functions

# ----------------------------------------------------------------------------------------------------------------------


def rleToMask(rle: str, shape: tuple = (1400, 2100)) -> np.ndarray:
    """
    Converting an RLE encoding to a mask

     Parameter
     ----------
     rle   : RLE encoding to convert
     shape : mask shape

     Return
     ----------
     np.array : mask
    """

    width, height = shape[:2]

    mask = np.zeros(width * height).astype(np.uint8)

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start + lengths[index])] = 1
        current_position += lengths[index]

    return mask.reshape(height, width).T


def maskInColor(image: np.ndarray,
                mask: np.ndarray,
                color: tuple = (0, 0, 255),
                alpha: float = 0.2) -> np.ndarray:
    """
    Overlay mask on image

     Parameter
     ----------
     image : image on which we want to overlay the mask
     mask  : mask to process
     color : color we want to apply on mask
     alpha : opacity coefficient

     Return
     ----------
     np.array : result of layering
    """

    image = np.array(image)
    H, W, C = image.shape
    mask = mask.reshape(H, W)
    overlay = image.astype(np.float32)
    color_arr = np.tile(color, (H, W, 1))
    mask_expanded = np.expand_dims(mask, axis=2)
    overlay = 255 - (255 - overlay) * (1 - mask_expanded * alpha * color_arr / 255)
    overlay = np.clip(overlay, 0, 255)
    overlay = overlay.astype(np.uint8)
    return overlay


def trace_boundingBox(image: np.ndarray,
                      mask: np.ndarray,
                      color: tuple = (0, 0, 255),
                      width: int = 10):
    """
    Draw a bounding box on image

     Parameter
     ----------
     image : image on which we want to draw the box
     mask  : mask to process
     color : color we want to use to draw the box edges
     width : box edges's width

    """

    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        coin1 = (prop.bbox[3], prop.bbox[2])
        coin2 = (prop.bbox[1], prop.bbox[0])
        cv2.rectangle(image, coin2, coin1, color, width)


def cloudInColor(image: np.ndarray,
                 mask: np.ndarray,
                 color: tuple = (0, 0, 255),
                 alpha: float = 0.7,
                 threshold: int = 90) -> np.ndarray:
    """
    Draw a bounding box on image

     Parameter
     ----------
     image : image on which we want to colorize parts
     mask  : mask to process
     color : color we want to use to colorize image
     alpha : opacity coefficient
     threshold : pixel value threshold to apply

     Return
     ----------
     np.array : result of layering
    """
    imZone = cv2.bitwise_and(image, image, mask=mask)
    image_gray = cv2.cvtColor(imZone, cv2.COLOR_RGB2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(image_gray,
                                                 threshold,
                                                 255,
                                                 cv2.THRESH_BINARY)
    return maskInColor(image, blackAndWhiteImage, color=color, alpha=alpha)


# ----------------------------------------------------------------------------------------------------------------------

#                                                  Data visualisation

# ----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    repBase = '/Volumes/Pierre_2TO/Cours/master/semestre 3/Machine_learning/Advanced-machine-learning-project'
    repTrain = r'D:\Cours\master\semestre 3\Machine_learning\Advanced-machine-learning-project\Data'

    # split image_label column into image column and label column
    df = pd.read_csv(os.path.join(repTrain, 'train.csv'))
    df['image'] = df['Image_Label'].apply(lambda x: x.split('_')[0])
    df['label'] = df['Image_Label'].apply(lambda x: x.split('_')[1])
    df = df.drop(['Image_Label'], axis=1)

    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0)]
    image_name = '043e76c.jpg'
        # '043e76c.jpg' # problem de recoupage de masks
    #image_name = '008233e.jpg'

    rles = df[df['image'] == image_name]['EncodedPixels'].reset_index(drop=True)
    image_start = plt.imread(os.path.join(repTrain, 'initial_data', 'train_images', image_name))

    # Raw image
    fig, ax = plt.subplots()
    ax.imshow(image_start)

    # Colored rectangles
    fig, ax = plt.subplots()

    image = np.copy(image_start)  # reset the working image
    for k in range(4):  # We have 4 classes in this dataset
        rle = rles[k]  # initialize the current RLE code
        if not isinstance(rle, float):  # it's not a 'NaN' RLE
            mask = rleToMask(rles[k])
            trace_boundingBox(image, mask, color=colors[k])

    ax.imshow(image)
    plt.show()


