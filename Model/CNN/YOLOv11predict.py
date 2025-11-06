from ultralytics import YOLO

def main():
    # load the best model from training
    yolobest = YOLO("runs/detect/train/weights/best.pt")

    # Set the source path for test images
    sourcePath = "datasets/AgroPest/test/images"

    # Perform prediction on the test images
    yolobest.predict(
        source=sourcePath,
        imgsz=960, # Set up Image size
        save=True, # Save the detection box
        save_txt=False, # Do not save the txt file for bounding box coordinates
        project="runs/detect", # Set up the save place for prediction
        device=0 # Use the first GPU
    )

if __name__ == "__main__":
    main()
