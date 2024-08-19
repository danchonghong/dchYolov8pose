from ultralytics import YOLO

# model = YOLO("yolov8n.pt")
model = YOLO("/home/user/ball/pytorch_keypoint/"
             "dch-Yolov8/ultralytics/models/yolo/pose/runs/pose/train47/weights/best.pt")  # 模型文件路径
"""
1nnnnn.jpg
11111.jpg
MP21.jpg
MP22.jpg
"""


results = model("1nnnnn.jpg", visualize=True)  # 要预测图片路径和使用可视化
