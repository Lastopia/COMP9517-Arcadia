from ultralytics import YOLO

#This file is to detect and classify pests in images from datasets\AgroPest using YOLOv11n model

def main():
#load a model
    yolo = YOLO("yolo11n.pt")

    #train the model
    yolo.train(
        data="data.yaml", 
        workers=0, # Set up for windows
        epochs=100, # Number of train iterations
        imgsz=960, # Image size
        batch=16, # Number of images be processed per iteration
        seed=0, # Make the result to be reproducible
        device=0 # Set up to use the first GPU
        )

if __name__ == "__main__":
    main()
