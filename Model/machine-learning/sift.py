import cv2
import numpy as np

def extract_sift_features(image, max_features=200):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=max_features)
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None or len(descriptors) == 0:
        return np.zeros(128, dtype=np.float32)

    sift_vector = np.mean(descriptors, axis=0)
    
    return sift_vector
