# Performance Evaluation of YOLOv5 and YOLOv8 on Curated Dataset
🔍 A deep learning-based object detection model for locust identification using YOLOv5 and YOLOv8.

📌 Project Overview

This project aims to detect locusts in images using YOLOv5 and YOLOv8 models. The dataset has been curated and annotated and augmentated specifically for this task.
This repository contains code for detecting locusts using YOLOv5 and YOLOv8. The dataset has been specifically curated for this task and can be used to train and evaluate the models.
This repository contains the implementations of YOLOv5 and YOLOv8 for locust detection. The objective of this study is to compare the performance of these two models in terms of detection accuracy and efficiency.

After extensive evaluation, YOLOv8 demonstrated superior performance compared to YOLOv5, achieving higher mAP (mean Average Precision) and better detection accuracy.

✅ Features
YOLOv5 & YOLOv8 implementations
Custom dataset(without augmentation and with augmentation) for locust detection
Training and inference scripts included
Supports Google Colab for easy execution

🔹 Dataset DOI: 10.5281/zenodo.14964987
🔹 YOLOv5 and YOLOv8 Implementations included

📁 Dataset & Model Files
The dataset is hosted on Zenodo at the DOI link above.
YOLOv5 & YOLOv8 models need to be trained using this dataset.

🚀 Installation & Setup

1️⃣ Clone the Repository

git clone https://github.com/yourusername/locust-detection.git
cd locust-detection

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Download the Dataset
Since the dataset is hosted on Zenodo, download it manually from:
🔗 Dataset DOI: 10.5281/zenodo.14964987

Unzip and place it in the correct directory (datasets/).

4️⃣ Mount Google Drive (if using Colab)
from google.colab import drive
drive.mount('/content/drive')


🔧 Training 

Train YOLOv5

!python train.py --img 640 --batch 16 --epochs 60 --data robo.dataset.yaml --weights yolov5s.pt --cache

Train YOLOv8

results = model.train(
    data=dataset_yaml_path,  # Use the correct path here
    epochs=60,
    batch=16,
    imgsz=640,
    save_period=10,
    project=save_model_path,
    name='yolov8_detection'
)

🏆 Inference

Inference YOLOv5

!python detect.py --weights /content/drive/MyDrive/locust_detection/yolov5/runs/train/exp/weights/last.pt \
                   --conf 0.40 --data dataset.yaml \
                   --source /content/drive/MyDrive/locust_detection/test_data/images

Inference YOLOv8

results = model.predict(source=source_directory, conf=0.40, save=True, project=output_directory, name='detection_results')

## 📸 Results
Below are sample detection results:

### **YOLOv5 Output**
![YOLOv5 Detection](results/yolov5_output.jpg)

### **YOLOv8 Output**
![YOLOv8 Detection](results/yolov8_output.jpg)


📜 Citation

If you use this dataset, please cite the Zenodo DOI:  
🔗 DOI: [10.5281/zenodo.14964987](https://doi.org/10.5281/zenodo.14964987)  


📄 License
🔹 This project is licensed under the MIT License.

🤝 Acknowledgments
Special thanks to Ultralytics for YOLO development.

📬 Contact

For questions, reach out via poojavjpy@gmail.com or https://www.researchgate.net/profile/Pooja-Vajpayee-2/research



