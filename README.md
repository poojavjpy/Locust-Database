# Performance Evaluation of YOLOv5 and YOLOv8 on Curated Dataset
🔍 A deep learning-based object detection model for locust identification using YOLOv5 and YOLOv8.

📌 Project Overview
This project aims to detect locusts in images using YOLOv5 and YOLOv8 models. The dataset has been curated and annotated specifically for this task.

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

📌 Training & Inference
Train YOLOv5

python train.py --img 640 --batch 16 --epochs 50 --data data.yaml --weights yolov5s.pt

