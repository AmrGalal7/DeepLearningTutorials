# YOLO v3 from Scratch Using Pytorch

### Reference:
  Almost the entire code is from:
  - [Implement YOLO v3 from scratch](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/)
  
### Topics:
  1. YOLO for object detection.
  2. PyTorch

### Running:
  --> make sure the architecture's configiration file (yolo.cfg) is in './config/', and the weights file is in the main directory.
  
  - For images:
    ```
    python detect.py --images filename --det det
    ```
  - For videos:
    ```
    python detect.py --videos filename --det det
    ```
  - For webcam:
    ```
    python detect.py --videos 0 --det det
    ```
  
### Other Resources/ Readings:
- [Real-time Object Detection with YOLO, YOLOv2 and now YOLOv3](https://medium.com/@jonathan_hui/real-time-object-detection-with-yolo-yolov2-28b1b93e2088)
