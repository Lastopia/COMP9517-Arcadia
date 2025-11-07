from skimage.feature import hog
import cv2
import numpy as np

def extract_hog_features(image,
                         orientations=9,
                         pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2),
                         resize_shape=(128, 128)):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, resize_shape)

    features = hog(gray,
                   orientations=orientations,
                   pixels_per_cell=pixels_per_cell,
                   cells_per_block=cells_per_block,
                   block_norm='L2-Hys',
                   visualize=False)
    
    # channel_axis= -1

    return np.array(features, dtype=np.float32)
