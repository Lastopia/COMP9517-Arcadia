# Insect Detection and Classification on AgroPest-12 Dataset
### GROUP: COMP9517-Arcadia 

## Introduction
This project presents the development and comparison of several traditional machine learning methods and deep learning models for pest detection and classification in the AgroPest-12 dataset.
The aim is to measure their performance to analyze strengths, weaknesses, computational cost, and explainability.

## Dataset
We use the AgroPest-12 dataset, containing 12 pest classes with annotated bounding boxes.
- **Source**：https://www.kaggle.com/datasets/rupankarmajumdar/crop-pests-dataset

## Team Contributions
`Chenyi Li:`
- YOLO - v11
  - **Source**：https://github.com/ultralytics/ultralytics
- Transformer - RT-DETR
  - **Source**：https://github.com/lyuwenyu/RT-DETR
- XAI for yolov11 & RT-DETR
  - **Source for yolov11 XAI**：https://github.com/rigvedrs/YOLO-V12-CAM
  - **Source for rtdetr XAI**：https://github.com/jacobgil/pytorch-grad-cam

`Yiyang Shen:`
- inceptionV3
- EfficentNet

`Yutong Zhang:`
- CNN - LeNet
- CNN - VGG
- CNN - ResNet
- CNN - Faster R-CNN

 `Yian Zhu:`
 - YOLOv8 + MobileNet

 `Ruiying Ren:`
 - SIFT/LBP
 - SVM/KNN

## Environment Setup
```bash
pip install ultralytics grad-cam opencv-python ttach
```

## References
@software{yolo11_ultralytics,
  author = {Glenn Jocher and Jing Qiu},
  title = {Ultralytics YOLO11},
  version = {11.0.0},
  year = {2024},
  url = {https://github.com/ultralytics/ultralytics},
  orcid = {0000-0001-5950-6979, 0000-0003-3783-7069},
  license = {AGPL-3.0}
}

@misc{lv2023detrs,
      title={DETRs Beat YOLOs on Real-time Object Detection},
      author={Wenyu Lv and Shangliang Xu and Yian Zhao and Guanzhong Wang and Jinman Wei and Cheng Cui and Yuning Du and Qingqing Dang and Yi Liu},
      year={2023},
      eprint={2304.08069},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{jacobgilpytorchcam,
  title={PyTorch library for CAM methods},
  author={Jacob Gildenblat and contributors},
  year={2021},
  publisher={GitHub},
  howpublished={\url{https://github.com/jacobgil/pytorch-grad-cam}},
}

The XAI for YOLOv11 of this project makes use of the YOLO-V12-CAM repository:
https://github.com/rigvedrs/YOLO-V12-CAM
