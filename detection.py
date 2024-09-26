import torch
import torchvision.transforms as T
import cv2
import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# โหลดโมเดล Faster R-CNN
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
num_classes = 2  # 1 คลาส + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
# โหลด weights เข้าไปในโมเดล
model_path = r'D:\ปี3\เทอม1\ปลายภาค\Computer Vision\miniProject\Faster-RCNN\Faster_R-CNN.pt'  # เปลี่ยนเป็น path ของไฟล์โมเดล
model.load_state_dict(torch.load(model_path))
model.eval()  # ตั้งค่าโมเดลให้เป็นโหมด eval

# ฟังก์ชันสำหรับการเตรียมรูปภาพ
def transform_image(image):
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((384, 384)),  # ปรับขนาดให้เข้ากับโมเดล
        T.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  # เพิ่มมิติ batch

# อ่านรูปภาพ
img_path = r'IVUS-100.v1i.tensorflow\valid\frame_01_0011_003_png.rf.be1255a41505f0ec06b80c6a69f47026.jpg'  # เปลี่ยนเป็น path ของรูปภาพที่ต้องการตรวจจับ
image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # เปลี่ยนสีจาก BGR เป็น RGB

# เตรียมรูปภาพ
input_tensor = transform_image(image_rgb)

# ทำการตรวจจับวัตถุ
with torch.no_grad():
    predictions = model(input_tensor)

# แสดงผลลัพธ์
boxes = predictions[0]['boxes'].cpu().numpy()
scores = predictions[0]['scores'].cpu().numpy()
labels = predictions[0]['labels'].cpu().numpy()

# วาด bounding boxes บนรูปภาพ
for box, score, label in zip(boxes, scores, labels):
    if score > 0.7:  # กำหนด threshold สำหรับการแสดง bounding box
        x1, y1, x2, y2 = box.astype(int)
        label = "Lumen" if label == 1 else "None"
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)  # วาดกรอบ
        cv2.putText(image, f'Class: {label}, conf: {score:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# แสดงผลลัพธ์
plt.imshow(image)
plt.axis('off')  # ปิดแกน
plt.show()