# Background Estimation for YOLO-based Static Object Recognition

This repository implements an **efficient background estimation scheme** for video sequences to improve **static object recognition** using YOLOv3. The method uses a **sliding window histogram-based approach** to compute dominant background pixels over frames and integrates YOLO for object detection. The repository also includes an **improvised RGB-based algorithm** and evaluation metrics like PSNR, SSIM, Precision, Recall, and F1-score.

---

## Features

- Histogram-based background estimation for grayscale and RGB frames.  
- YOLOv3 integration for detecting objects on frames and estimated backgrounds.  
- Sliding window approach for adaptive background updating.  
- Evaluation using CDNet2014 dataset with ground truth comparisons.  
- Metrics: PSNR, SSIM, Precision, Recall, F1-score.  
- Supports custom video input for background detection.

---

## Dataset and Model Downloads

### CDNet2014 Dataset
- Kaggle: [CDNet2014](https://www.kaggle.com/datasets/maamri95/cdnet2014)  
- Google Drive: [Archive.zip](https://drive.google.com/drive/folders/1nhLOLUpfUIZFIzi29aNueD7zo6ToJ7iA?usp=sharing)  

**Folder structure after extraction:**  
archive/dataset/dynamicBackground/fall/input

bash
Copy
Edit

### YOLOv3 Model
- Linux:  
```bash
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
wget https://pjreddie.com/media/files/yolov3.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
Windows: Google Drive folder

Place the files in the src folder before running the code.

Folder Structure
bash
Copy
Edit
project-root/
│
├── archive/                # CDNet2014 dataset
├── data_video_frames/      # Custom video frames
├── src/                    # Python scripts
├── results/                # Output backgrounds and detections
├── yolov3.cfg              # YOLO configuration
├── yolov3.weights          # YOLO pre-trained weights
├── coco.names              # COCO class labels
└── README.md
Usage
1. Background Estimation on Custom Videos
python
Copy
Edit
# Set your video path
video_path = "videoblocks-high-timelapse-of-diverse-activity.mp4"
output_folder = "data_video_frames/"
Extract frames and save to input folder.

Run estimate_background_sliding_window() for grayscale or RGB backgrounds.

YOLO detection results saved in detections/ and boxes/.

2. Using CDNet Dataset
python
Copy
Edit
dataset_path = "archive/"
results_path = "archive/results/"
sequences = ["dataset/dynamicBackground/Fall"]
Run process_all_sequences() to generate background images and YOLO detection results.

Evaluation
Metrics include Precision, Recall, F1-score, PSNR, SSIM.

Evaluation compares YOLO on input frames vs YOLO on estimated backgrounds.

Ground truth from CDNet2014 dataset used for comparison.

Requirements
Python 3.8+

OpenCV (opencv-python)

NumPy

Torch (torch)

Pandas (pandas)

scikit-image (scikit-image)

Install dependencies using:

bash
Copy
Edit
pip install opencv-python torch numpy pandas scikit-image
References
Original Paper: An Efficient Scheme to Obtain Background Image in Video for YOLO-based Static Object Recognition

