import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from datetime import datetime
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
    plt.show()


def extract_features(processed_img): 
    X = [] 
    Y = [] 
    for (id, img) in tqdm(processed_img, desc="Extracting features", unit="img"): 
        sift_descriptors, kps = extract_sift_features(img, 150)

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) 
        mask = cv2.inRange(hsv, (0, 70, 50), (180, 255, 255)) 
        # keep only mid-bright objects
        masked = cv2.bitwise_and(img, img, mask=mask) 
        hog_descriptors = extract_hog_features(masked) 
        
        descriptors = np.hstack((sift_descriptors, hog_descriptors)) 
        X.append(descriptors) 
        Y.append(id) 

    return np.array(X), np.array(Y)

if __name__ == "__main__":
    image_path = "../../Data/train/images"
    label_path = "../../Data/train/labels"

    print("Reading and preprocessing training images...")
    processed_img = read_images(image_path, label_path, apply_preprocess=True, output_rgb=True)

    labels = np.array([cls for cls, _ in processed_img])

    # Use 50% of dataset for convenience
    _, sampled_img, _, _ = train_test_split(
        processed_img, labels,
        test_size=0.5,
        stratify=labels,
        random_state=42
    )

    print("Extracting features from training set...")
    X, Y = extract_features(sampled_img)

    print(f"{len(X)}Freature Extracted")

    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

    clf = train_svm(X, Y, tune=False)

    print("SVM TRAINED")

    test_image_path = "../../Data/test/images"
    test_label_path = "../../Data/test/labels"

    processed_test_img = read_images(test_image_path, test_label_path, apply_preprocess=True, output_rgb=True)

    X_test, Y_test = extract_features(processed_test_img)

    pre_label = evaluate_svm(clf, X_test)

    # print(np.unique(y_train, return_counts=True))
    
    print(np.unique(pre_label, return_counts=True))
    print(np.unique(Y_test, return_counts=True))

    with open('report', 'a+') as file:
        now = datetime.now()
        file.write(f"{now.strftime('%Y-%m-%d %H:%M:%S')} {accuracy_score(pre_label, Y_test):.4f}\n")

    evaluate_model("Method 1", Y_test, pre_label)
