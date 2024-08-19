from ultralytics import YOLO
from ultralytics.nn.tasks import PoseModel

# 加载训练好的模型或者网络结构配置文件
# model = YOLO('best.pt')
# model = YOLO('ultralytics/cfg/models/v8/yolov8n.yaml')yolov8-C2f-RVB.yaml
PoseModel('yolov8m-star.yaml')
model = YOLO('yolov8m-star.yaml')
print(model.info())