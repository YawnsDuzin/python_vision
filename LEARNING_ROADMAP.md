# Computer Vision & Deep Learning 학습 로드맵

공사현장 안전관리를 위한 단계별 학습 계획입니다.

---

## 단계 1: Python 기초 및 데이터 처리 (1-2주)

### 학습 목표
- Python 프로그래밍 기초 마스터
- 데이터 처리 및 시각화 능력 확보

### 세부 내용

#### 1.1 Python 기초
- **변수, 데이터 타입, 연산자**
  - 숫자, 문자열, 리스트, 튜플, 딕셔너리
  - 조건문, 반복문, 함수

- **객체지향 프로그래밍**
  - 클래스와 객체
  - 상속, 캡슐화, 다형성

- **파일 입출력 및 예외 처리**
  - 파일 읽기/쓰기
  - try-except 구문

#### 1.2 NumPy
```python
import numpy as np

# 배열 생성 및 조작
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.shape)  # (2, 3)

# 배열 연산
arr_squared = arr ** 2
arr_mean = np.mean(arr)

# 이미지 데이터는 NumPy 배열로 표현됨
# 예: (높이, 너비, 채널) = (1080, 1920, 3)
```

#### 1.3 Pandas
```python
import pandas as pd

# 데이터프레임 생성
df = pd.DataFrame({
    'worker_id': [1, 2, 3],
    'helmet': [True, False, True],
    'vest': [True, True, False]
})

# 데이터 분석
violation_rate = df[df['helmet'] == False].shape[0] / len(df)
```

#### 1.4 Matplotlib & Seaborn
```python
import matplotlib.pyplot as plt
import seaborn as sns

# 이미지 시각화
plt.imshow(image)
plt.title('Construction Site')
plt.show()

# 통계 시각화
sns.barplot(x='date', y='violations', data=safety_data)
```

### 실습 프로젝트
1. **데이터 분석**: 공사현장 안전 위반 데이터 분석
2. **시각화**: 일별/월별 안전 통계 대시보드 생성

### 추천 자료
- [Python 공식 문서](https://docs.python.org/3/)
- [NumPy 튜토리얼](https://numpy.org/doc/stable/user/quickstart.html)
- [Pandas 공식 가이드](https://pandas.pydata.org/docs/getting_started/index.html)

---

## 단계 2: OpenCV 및 이미지 처리 (2-3주)

### 학습 목표
- 이미지 처리 기본 개념 이해
- OpenCV를 활용한 이미지/비디오 처리 능력 확보

### 세부 내용

#### 2.1 OpenCV 기초
```python
import cv2

# 이미지 읽기 및 표시
image = cv2.imread('construction_site.jpg')
cv2.imshow('Site', image)
cv2.waitKey(0)

# 이미지 크기 조정
resized = cv2.resize(image, (640, 480))

# 색상 공간 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
```

#### 2.2 이미지 전처리
```python
# 노이즈 제거
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# 엣지 검출
edges = cv2.Canny(gray, 50, 150)

# 이진화
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 모폴로지 연산
kernel = np.ones((5,5), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
```

#### 2.3 색상 기반 객체 검출
```python
# HSV 범위로 안전조끼 검출 (주황색)
lower_orange = np.array([5, 100, 100])
upper_orange = np.array([15, 255, 255])

mask = cv2.inRange(hsv, lower_orange, upper_orange)
result = cv2.bitwise_and(image, image, mask=mask)
```

#### 2.4 컨투어 검출
```python
# 윤곽선 찾기
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 윤곽선 그리기
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# 바운딩 박스
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
```

#### 2.5 비디오 처리
```python
# 비디오 캡처
cap = cv2.VideoCapture('construction_site.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 처리
    processed = process_frame(frame)

    cv2.imshow('Frame', processed)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 실습 프로젝트
1. **안전조끼 검출기**: 색상 기반 안전조끼 착용 여부 확인
2. **위험 구역 표시**: ROI(관심 영역) 설정 및 침입 감지
3. **모션 감지**: 배경 차분법으로 움직임 감지

### 추천 자료
- [OpenCV 공식 문서](https://docs.opencv.org/)
- [OpenCV Python 튜토리얼](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

---

## 단계 3: 딥러닝 기초 (3-4주)

### 학습 목표
- 딥러닝 기본 개념 이해
- PyTorch 또는 TensorFlow 프레임워크 활용 능력 확보

### 세부 내용

#### 3.1 딥러닝 이론
- **인공신경망 (ANN)**
  - 퍼셉트론, 활성화 함수
  - 순전파, 역전파
  - 경사 하강법

- **CNN (Convolutional Neural Network)**
  - 합성곱 레이어 (Convolution)
  - 풀링 레이어 (Pooling)
  - 완전 연결 레이어 (Fully Connected)

- **학습 기법**
  - 손실 함수 (Loss Function)
  - 최적화 알고리즘 (Optimizer)
  - 정규화 (Regularization)
  - 배치 정규화 (Batch Normalization)
  - 드롭아웃 (Dropout)

#### 3.2 PyTorch 기초
```python
import torch
import torch.nn as nn
import torch.optim as optim

# 텐서 생성
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
y = torch.tensor([[5], [6]], dtype=torch.float32)

# 간단한 신경망
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 모델 학습
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
```

#### 3.3 이미지 분류 (CNN)
```python
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# 데이터셋 로드
dataset = ImageFolder('data/safety_images', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# CNN 모델
class SafetyClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(SafetyClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

#### 3.4 전이 학습 (Transfer Learning)
```python
import torchvision.models as models

# 사전 학습된 모델 로드
model = models.resnet50(pretrained=True)

# 마지막 레이어 교체 (안전모 착용 여부 분류: 2클래스)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# 일부 레이어 고정
for param in model.parameters():
    param.requires_grad = False

# 마지막 레이어만 학습
for param in model.fc.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
```

#### 3.5 모델 평가
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 평가 모드
model.eval()

predictions = []
targets = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())
        targets.extend(labels.cpu().numpy())

# 메트릭 계산
accuracy = accuracy_score(targets, predictions)
precision = precision_score(targets, predictions, average='weighted')
recall = recall_score(targets, predictions, average='weighted')
f1 = f1_score(targets, predictions, average='weighted')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')
```

### 실습 프로젝트
1. **안전모 분류기**: 이미지 분류로 안전모 착용 여부 판단
2. **안전장비 분류**: 다중 클래스 분류 (안전모, 안전조끼, 안전화 등)
3. **전이 학습 실습**: ResNet, VGG 등 사전 학습 모델 활용

### 추천 자료
- [PyTorch 공식 튜토리얼](https://pytorch.org/tutorials/)
- [Deep Learning Book (Ian Goodfellow)](https://www.deeplearningbook.org/)
- [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)

---

## 단계 4: Object Detection & Segmentation (4-6주)

### 학습 목표
- 객체 탐지 및 세그멘테이션 기술 습득
- 실시간 객체 탐지 시스템 구현 능력 확보

### 세부 내용

#### 4.1 Object Detection 이론
- **Two-Stage Detectors**
  - R-CNN, Fast R-CNN, Faster R-CNN
  - Region Proposal Network (RPN)

- **One-Stage Detectors**
  - YOLO (You Only Look Once)
  - SSD (Single Shot Detector)
  - RetinaNet

- **평가 지표**
  - IoU (Intersection over Union)
  - mAP (mean Average Precision)
  - Precision-Recall 곡선

#### 4.2 YOLO 실습
```python
# YOLOv8 사용 예시 (Ultralytics)
from ultralytics import YOLO

# 모델 로드
model = YOLO('yolov8n.pt')  # nano 모델

# 커스텀 데이터로 학습
model.train(
    data='safety_detection.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='safety_detector'
)

# 추론
results = model('construction_site.jpg')

# 결과 시각화
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        cls = box.cls[0]
        print(f'Class: {cls}, Confidence: {conf:.2f}')
```

#### 4.3 데이터셋 구성
```yaml
# safety_detection.yaml
train: ./datasets/safety/train/images
val: ./datasets/safety/val/images
test: ./datasets/safety/test/images

nc: 5  # 클래스 개수
names: ['person', 'helmet', 'vest', 'no-helmet', 'no-vest']
```

#### 4.4 데이터 어노테이션
```python
# YOLO 형식 레이블 생성
# 각 이미지에 대응하는 .txt 파일
# 형식: <class_id> <x_center> <y_center> <width> <height>
# 모든 값은 이미지 크기로 정규화 (0-1)

# 예시: person.txt
# 0 0.5 0.5 0.3 0.6
# 1 0.45 0.3 0.1 0.1
```

**추천 어노테이션 도구:**
- [Roboflow](https://roboflow.com/) - 웹 기반, 자동 증강
- [LabelImg](https://github.com/heartexlabs/labelImg) - 데스크톱 앱
- [CVAT](https://github.com/opencv/cvat) - 오픈소스, 협업 가능

#### 4.5 모델 최적화
```python
# 모델 양자화 및 경량화
model.export(format='onnx')  # ONNX 형식으로 변환
model.export(format='tflite')  # TensorFlow Lite

# TensorRT 최적화 (NVIDIA GPU)
model.export(format='engine', half=True)  # FP16 정밀도

# 추론 속도 향상
results = model('image.jpg', device='cuda:0', half=True)
```

#### 4.6 Instance Segmentation
```python
# Mask R-CNN 또는 YOLOv8-seg
from ultralytics import YOLO

# 세그멘테이션 모델
seg_model = YOLO('yolov8n-seg.pt')

# 학습
seg_model.train(
    data='safety_segmentation.yaml',
    epochs=100,
    imgsz=640
)

# 추론
results = seg_model('construction_site.jpg')

# 마스크 시각화
for result in results:
    masks = result.masks  # 세그멘테이션 마스크
    boxes = result.boxes  # 바운딩 박스
```

#### 4.7 실시간 탐지 시스템
```python
import cv2
from ultralytics import YOLO

model = YOLO('best.pt')

cap = cv2.VideoCapture(0)  # 웹캠
# cap = cv2.VideoCapture('rtsp://camera_ip/stream')  # IP 카메라

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 추론
    results = model(frame, conf=0.5)

    # 결과 그리기
    annotated_frame = results[0].plot()

    # 안전 위반 감지
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        if model.names[cls_id] == 'no-helmet':
            # 알림 발생
            print('ALERT: Worker without helmet detected!')

    cv2.imshow('Safety Monitor', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 실습 프로젝트
1. **PPE 탐지기**: 안전모, 안전조끼 착용 여부 실시간 탐지
2. **차량 및 장비 추적**: 중장비 및 차량 탐지 및 추적
3. **위험 구역 모니터링**: ROI 내 사람 침입 감지
4. **세그멘테이션 기반 분석**: 작업 영역 세분화 및 분석

### 추천 자료
- [Ultralytics YOLOv8 문서](https://docs.ultralytics.com/)
- [Detectron2 (Facebook AI)](https://github.com/facebookresearch/detectron2)
- [MMDetection](https://github.com/open-mmlab/mmdetection)

---

## 단계 5: 공사현장 안전관리 프로젝트 (6-8주)

### 학습 목표
- 실무 수준의 안전관리 시스템 구축
- 엔드투엔드 프로젝트 완성

### 세부 프로젝트

#### 5.1 통합 안전 모니터링 시스템

**기능:**
- 다중 카메라 지원
- 실시간 PPE 탐지
- 위험 구역 침입 감지
- 자동 알림 시스템
- 통계 대시보드

**아키텍처:**
```
cameras/ (RTSP 스트림)
    ↓
preprocessing.py (프레임 수집 및 전처리)
    ↓
detection.py (YOLO 모델 추론)
    ↓
tracking.py (객체 추적 - DeepSORT)
    ↓
rules_engine.py (규칙 기반 위반 감지)
    ↓
alert_system.py (알림 발송 - 이메일/SMS)
    ↓
dashboard.py (Streamlit/Flask 대시보드)
    ↓
database.py (SQLite/PostgreSQL 저장)
```

**구현 예시:**
```python
# main.py
import cv2
from detection import SafetyDetector
from tracking import ObjectTracker
from alert_system import AlertManager
from database import SafetyDB

class SafetyMonitoringSystem:
    def __init__(self):
        self.detector = SafetyDetector('models/safety_detector.pt')
        self.tracker = ObjectTracker()
        self.alert_manager = AlertManager()
        self.db = SafetyDB('safety_records.db')

    def process_camera(self, camera_id, rtsp_url):
        cap = cv2.VideoCapture(rtsp_url)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 탐지
            detections = self.detector.detect(frame)

            # 추적
            tracked_objects = self.tracker.update(detections)

            # 규칙 적용
            violations = self.check_violations(tracked_objects)

            # 위반 처리
            for violation in violations:
                self.alert_manager.send_alert(violation)
                self.db.log_violation(camera_id, violation)

            # 시각화
            annotated = self.visualize(frame, tracked_objects, violations)

            yield annotated

    def check_violations(self, objects):
        violations = []

        for obj in objects:
            # PPE 미착용 체크
            if obj['type'] == 'person':
                if not obj.get('has_helmet'):
                    violations.append({
                        'type': 'NO_HELMET',
                        'object_id': obj['id'],
                        'timestamp': obj['timestamp'],
                        'location': obj['bbox']
                    })

                if not obj.get('has_vest'):
                    violations.append({
                        'type': 'NO_VEST',
                        'object_id': obj['id'],
                        'timestamp': obj['timestamp'],
                        'location': obj['bbox']
                    })

            # 위험 구역 체크
            if self.in_danger_zone(obj['bbox']):
                violations.append({
                    'type': 'DANGER_ZONE',
                    'object_id': obj['id'],
                    'timestamp': obj['timestamp'],
                    'location': obj['bbox']
                })

        return violations
```

#### 5.2 행동 인식 시스템

**기능:**
- 위험한 작업 자세 감지
- 낙상 감지
- 이상 행동 탐지

**기술 스택:**
- Pose Estimation (MediaPipe, OpenPose)
- Action Recognition (I3D, SlowFast)

```python
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def analyze_worker_pose(frame):
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # 허리 굽힘 각도 계산
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]

        angle = calculate_angle(shoulder, hip, knee)

        if angle < 90:  # 허리를 과도하게 굽힘
            return 'UNSAFE_POSTURE'

    return 'SAFE'
```

#### 5.3 데이터 분석 및 리포팅

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class SafetyAnalytics:
    def __init__(self, db_path):
        self.db = SafetyDB(db_path)

    def generate_daily_report(self, date):
        violations = self.db.get_violations_by_date(date)
        df = pd.DataFrame(violations)

        # 위반 유형별 통계
        violation_counts = df['type'].value_counts()

        # 시간대별 분석
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        hourly_violations = df.groupby('hour').size()

        # 구역별 분석
        zone_violations = df.groupby('camera_id').size()

        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        violation_counts.plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Violations by Type')

        hourly_violations.plot(kind='line', ax=axes[0, 1])
        axes[0, 1].set_title('Hourly Violations')

        zone_violations.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Violations by Zone')

        # 히트맵
        pivot = df.pivot_table(
            values='id',
            index='hour',
            columns='type',
            aggfunc='count'
        )
        sns.heatmap(pivot, annot=True, fmt='g', ax=axes[1, 1])
        axes[1, 1].set_title('Violation Heatmap')

        plt.tight_layout()
        plt.savefig(f'reports/safety_report_{date}.png')

        return {
            'total_violations': len(df),
            'by_type': violation_counts.to_dict(),
            'peak_hour': hourly_violations.idxmax(),
            'most_dangerous_zone': zone_violations.idxmax()
        }
```

#### 5.4 웹 대시보드 (Streamlit)

```python
import streamlit as st
import cv2

st.title('🚧 Construction Safety Monitoring System')

# 사이드바
st.sidebar.header('Settings')
camera_id = st.sidebar.selectbox('Select Camera', ['CAM-01', 'CAM-02', 'CAM-03'])
conf_threshold = st.sidebar.slider('Confidence Threshold', 0.0, 1.0, 0.5)

# 실시간 스트림
col1, col2 = st.columns(2)

with col1:
    st.subheader('Live Feed')
    video_placeholder = st.empty()

with col2:
    st.subheader('Statistics')
    stats_placeholder = st.empty()

# 시스템 실행
system = SafetyMonitoringSystem()

for frame in system.process_camera(camera_id, CAMERA_URLS[camera_id]):
    video_placeholder.image(frame, channels='BGR')

    # 통계 업데이트
    stats = system.get_current_stats()
    stats_placeholder.json(stats)

# 일일 리포트
st.subheader('Daily Report')
date = st.date_input('Select Date')

if st.button('Generate Report'):
    analytics = SafetyAnalytics('safety_records.db')
    report = analytics.generate_daily_report(date)

    st.json(report)
    st.image(f'reports/safety_report_{date}.png')
```

### 배포 및 운영

#### Docker 컨테이너화
```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0

WORKDIR /app

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .

CMD ["python3", "main.py"]
```

#### 클라우드 배포
- **AWS EC2**: GPU 인스턴스 (p3.2xlarge)
- **Google Cloud**: Compute Engine with GPU
- **Azure**: VM with NVIDIA GPU

#### 엣지 디바이스
- **NVIDIA Jetson**: Nano, Xavier, Orin
- **Intel NUC**: Neural Compute Stick
- **Raspberry Pi**: (경량 모델만)

### 추천 자료
- [Streamlit 문서](https://docs.streamlit.io/)
- [Docker 공식 가이드](https://docs.docker.com/)
- [NVIDIA Jetson 개발자 가이드](https://developer.nvidia.com/embedded/jetson)

---

## 추가 학습 주제

### 고급 기술
1. **Multi-Object Tracking (MOT)**
   - DeepSORT, ByteTrack
   - Re-identification

2. **3D Vision**
   - Depth Estimation
   - 3D Object Detection
   - Point Cloud Processing

3. **Video Understanding**
   - Temporal Action Detection
   - Video Summarization

4. **Edge AI**
   - Model Quantization
   - Pruning & Distillation
   - ONNX Runtime

### 도메인 특화
1. **건설 안전 규정**
   - OSHA 기준
   - 국내 산업안전보건법

2. **카메라 네트워크**
   - RTSP/ONVIF 프로토콜
   - 카메라 캘리브레이션

3. **시스템 통합**
   - BIM (Building Information Modeling) 연동
   - IoT 센서 통합

---

## 학습 팁

1. **프로젝트 중심 학습**
   - 이론과 실습을 병행
   - 작은 프로젝트부터 시작해서 점진적으로 확장

2. **코드 리뷰**
   - GitHub에서 유명 프로젝트 코드 분석
   - 오픈소스 기여

3. **커뮤니티 활용**
   - Kaggle 대회 참여
   - Papers with Code
   - Reddit (r/computervision, r/MachineLearning)

4. **논문 읽기**
   - arXiv에서 최신 논문 구독
   - 주요 컨퍼런스: CVPR, ICCV, ECCV, NeurIPS

5. **실험 및 기록**
   - Jupyter Notebook으로 실험 기록
   - Weights & Biases, MLflow로 실험 관리

---

## 체크리스트

### 단계 1 완료 조건
- [ ] Python 기초 문법 숙지
- [ ] NumPy 배열 조작 능력
- [ ] Pandas 데이터프레임 다루기
- [ ] Matplotlib으로 시각화
- [ ] 간단한 데이터 분석 프로젝트 완성

### 단계 2 완료 조건
- [ ] OpenCV로 이미지 읽기/쓰기
- [ ] 색상 공간 변환
- [ ] 이미지 전처리 기법 적용
- [ ] 비디오 처리
- [ ] 색상 기반 객체 검출 프로젝트 완성

### 단계 3 완료 조건
- [ ] CNN 구조 이해
- [ ] PyTorch/TensorFlow 기본 사용
- [ ] 이미지 분류 모델 학습
- [ ] 전이 학습 적용
- [ ] 모델 평가 및 개선

### 단계 4 완료 조건
- [ ] YOLO 모델 학습 및 추론
- [ ] 커스텀 데이터셋 구축
- [ ] 실시간 객체 탐지 구현
- [ ] 모델 최적화
- [ ] PPE 탐지 시스템 구축

### 단계 5 완료 조건
- [ ] 다중 카메라 통합 시스템
- [ ] 실시간 알림 시스템
- [ ] 데이터베이스 연동
- [ ] 웹 대시보드 구축
- [ ] 배포 및 운영 경험

---

## 타임라인 (8주 집중 과정)

| 주차 | 학습 내용 | 실습 프로젝트 |
|------|-----------|--------------|
| 1주차 | Python 기초, NumPy, Pandas | 안전 데이터 분석 |
| 2주차 | OpenCV 기초, 이미지 처리 | 안전조끼 검출기 |
| 3주차 | 딥러닝 이론, PyTorch | 안전모 분류기 |
| 4주차 | CNN, 전이 학습 | 다중 클래스 분류 |
| 5주차 | Object Detection, YOLO | PPE 탐지 시스템 |
| 6주차 | 실시간 탐지, 추적 | 통합 모니터링 시스템 |
| 7주차 | 행동 인식, 고급 기능 | 위험 행동 감지 |
| 8주차 | 웹 대시보드, 배포 | 완성된 시스템 배포 |

**목표: 8주 후 실무에서 사용 가능한 공사현장 안전관리 시스템 구축**

---

## 다음 단계

학습 로드맵을 따라 진행하면서:
1. 각 단계별 실습 프로젝트 완성
2. GitHub에 코드 정리 및 업로드
3. 포트폴리오 구축
4. 실제 공사현장 데이터 확보 (가능한 경우)
5. 시스템 고도화 및 최적화

**성공을 기원합니다! 🚀**
