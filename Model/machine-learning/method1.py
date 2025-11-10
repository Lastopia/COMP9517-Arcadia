import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sift import extract_sift_features
from hog import extract_hog_features
from svm import train_svm, evaluate_svm
from preprocessing import read_images  # reuse your preprocessing
from sklearn.metrics import (accuracy_score, 
                            precision_score,
                            recall_score,
                            f1_score,
                            classification_report, 
                            confusion_matrix,
                            ConfusionMatrixDisplay)

# A helper function for evaluation
def evaluate_model(name, real_label, predicted_label):
    print(f"\n=== {name} Performance ===")
    print(f"Accuracy : {accuracy_score(real_label, predicted_label):.4f}")
    print(f"Precision: {precision_score(real_label, predicted_label, average='macro'):.4f}")
    print(f"Recall   : {recall_score(real_label, predicted_label, average='macro'):.4f}")
    print(f"F1-score : {f1_score(real_label, predicted_label, average='macro'):.4f}")

    print("\nClassification report:\n", classification_report(real_label, predicted_label))
    
    cm = confusion_matrix(real_label, predicted_label)

    print("ConfusionMatrix:")

    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=np.unique(real_label))
    cm_display.plot()
    # plt.show()


def extract_features(processed_img): 
    X = [] 
    Y = [] 
    for (id, img) in processed_img: 
        sift_descriptors = extract_sift_features(img, 150) 

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) 
        mask = cv2.inRange(hsv, (0, 30, 50), (180, 255, 255)) 
        # keep only mid-bright objects
        masked = cv2.bitwise_and(img, img, mask=mask) 
        hog_descriptors = extract_hog_features(masked) 
        
        descriptors = np.hstack((sift_descriptors, hog_descriptors)) 
        X.append(descriptors) 
        Y.append(id) 
        
    return np.array(X), np.array(Y)

if __name__ == "__main__":
    image_path = "../../Data/test/images"
    label_path = "../../Data/test/labels"

    processed_img = read_images(image_path, label_path, apply_preprocess=True, output_rgb=True)

    X, Y = extract_features(processed_img)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

    clf = train_svm(X_train, y_train, tune=False)

    pre_label = evaluate_svm(clf, X_test)
    
    print(np.unique(pre_label, return_counts=True))
    print(np.unique(y_test, return_counts=True))

    now = datetime.now()

    with open('report', 'a+') as file:
        file.write(f"{now.strftime('%Y-%m-%d %H:%M:%S')} {accuracy_score(pre_label, y_test):.4f}\n")

    evaluate_model("Method 1", y_test, pre_label)

