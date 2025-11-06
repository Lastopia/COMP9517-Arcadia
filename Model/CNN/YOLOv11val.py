from ultralytics import YOLO

def main():
    # load the best model from training
    yolobest = YOLO("runs/detect/train/weights/best.pt")

    # validate the model
    yoloval = yolobest.val(
        data="data.yaml", 
        imgsz=960, # Set up image size
        device=0, # Use the first GPU
        workers=0 # Set up for Windows
        )

if __name__ == "__main__":
    main()
