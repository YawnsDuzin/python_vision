# 공사현장 안전관리 레퍼런스 및 개발 가이드

건설 현장 안전관리를 위한 Computer Vision 적용 사례, 데이터셋, 모델, 프로젝트 레퍼런스입니다.

---

## 목차
1. [주요 적용 분야](#주요-적용-분야)
2. [오픈소스 데이터셋](#오픈소스-데이터셋)
3. [사전 학습 모델](#사전-학습-모델)
4. [오픈소스 프로젝트](#오픈소스-프로젝트)
5. [연구 논문](#연구-논문)
6. [상용 솔루션](#상용-솔루션)
7. [개발 방법론](#개발-방법론)
8. [규제 및 표준](#규제-및-표준)

---

## 주요 적용 분야

### 1. 개인보호구(PPE) 탐지

**감지 대상:**
- 안전모 (Hard Hat)
- 안전조끼 (Safety Vest)
- 안전화 (Safety Boots)
- 안전 고글 (Safety Goggles)
- 안전장갑 (Safety Gloves)
- 귀마개/귀덮개 (Ear Protection)
- 안전벨트 (Safety Harness)

**기술 스택:**
- Object Detection: YOLOv8, Faster R-CNN
- Classification: ResNet, EfficientNet
- 실시간 처리: TensorRT, ONNX Runtime

### 2. 위험 구역 관리

**주요 기능:**
- ROI (Region of Interest) 설정
- 출입 통제 구역 모니터링
- 중장비 작업 반경 감지
- 고소 작업 구역 관리

**구현 방법:**
```python
import cv2
import numpy as np

# 위험 구역 정의 (다각형)
danger_zone = np.array([
    [100, 200],
    [300, 200],
    [300, 400],
    [100, 400]
], dtype=np.int32)

def is_in_danger_zone(bbox, zone):
    """
    바운딩 박스 중심점이 위험 구역 내부인지 확인
    """
    x_center = (bbox[0] + bbox[2]) / 2
    y_center = (bbox[1] + bbox[3]) / 2
    point = (int(x_center), int(y_center))

    result = cv2.pointPolygonTest(zone, point, False)
    return result >= 0

# 시각화
def draw_danger_zone(frame, zone):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [zone], (0, 0, 255))  # 빨간색
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    cv2.polylines(frame, [zone], True, (0, 0, 255), 2)
    return frame
```

### 3. 작업자 행동 분석

**감지 행동:**
- 낙상 (Fall Detection)
- 위험한 작업 자세 (Unsafe Posture)
- 무단 출입 (Unauthorized Access)
- 비정상적 움직임 (Abnormal Movement)

**기술:**
- Pose Estimation: MediaPipe, OpenPose, AlphaPose
- Action Recognition: SlowFast, I3D, TSN
- Anomaly Detection: AutoEncoder, One-Class SVM

### 4. 차량 및 장비 관리

**모니터링 대상:**
- 굴삭기 (Excavator)
- 크레인 (Crane)
- 덤프트럭 (Dump Truck)
- 지게차 (Forklift)
- 믹서트럭 (Concrete Mixer)

**추적 기술:**
- Multi-Object Tracking: DeepSORT, ByteTrack
- Speed Estimation: Optical Flow
- Path Prediction: Kalman Filter

### 5. 환경 모니터링

**감지 항목:**
- 화재 및 연기 감지
- 붕괴 위험 감지
- 침수 감지
- 먼지 농도 측정 (IoT 센서 연동)

---

## 오픈소스 데이터셋

### 1. 건설 안전 관련 데이터셋

#### CHVG (Construction Hazard Video Dataset)
- **내용**: 건설 현장 위험 상황 비디오
- **규모**: 1,000+ 비디오 클립
- **레이블**: 12가지 위험 유형
- **링크**: [GitHub - CHVG](https://github.com/Yuchenliu98/CHVG)

#### Hard Hat Workers Dataset
- **내용**: 안전모 착용 여부 이미지
- **규모**: 5,000+ 이미지
- **클래스**: `helmet`, `no-helmet`, `person`
- **다운로드**: [Kaggle - Hard Hat Detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)
- **형식**: PASCAL VOC, YOLO

#### Construction Site Safety Image Dataset
- **내용**: 건설 현장 안전장비 착용 여부
- **규모**: 8,000+ 이미지
- **클래스**: `helmet`, `vest`, `mask`, `gloves`, `goggles`
- **다운로드**: [Roboflow - Construction Safety](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety)

#### SODA (Site Object Detection Dataset)
- **내용**: 건설 현장 객체 탐지
- **규모**: 15,000+ 이미지
- **클래스**: 20+ 객체 (작업자, 장비, 차량 등)
- **특징**: 다양한 날씨 및 조명 조건

### 2. 일반 PPE 데이터셋

#### PPE Detection Dataset
- **링크**: [GitHub - PPE Detection](https://github.com/ciber-lab/pictor-ppe)
- **규모**: 3,000+ 이미지
- **클래스**: 5가지 PPE 유형

#### Safety Helmet Wearing Dataset
- **링크**: [GitHub - SHWD](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset)
- **규모**: 7,581 이미지
- **특징**: 다양한 각도 및 조명

### 3. 행동 인식 데이터셋

#### UCF-101 (일반 행동, 건설 관련 일부 포함)
- **링크**: [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php)
- **규모**: 13,320 비디오
- **클래스**: 101가지 행동

#### Kinetics-700 (일반 행동)
- **링크**: [DeepMind Kinetics](https://www.deepmind.com/open-source/kinetics)
- **규모**: 650,000+ 비디오

### 4. 데이터셋 생성 도구

#### Roboflow
- **기능**: 어노테이션, 증강, 형식 변환
- **링크**: https://roboflow.com/
- **무료 플랜**: 가능 (제한적)

#### CVAT (Computer Vision Annotation Tool)
- **기능**: 이미지/비디오 어노테이션
- **링크**: https://github.com/opencv/cvat
- **설치**: 로컬 또는 클라우드

#### Label Studio
- **기능**: 다목적 어노테이션 도구
- **링크**: https://labelstud.io/

---

## 사전 학습 모델

### 1. YOLOv8 기반 모델

#### YOLOv8-PPE (커스텀 학습)
```python
from ultralytics import YOLO

# 사전 학습된 모델 다운로드 (예시)
model = YOLO('yolov8n.pt')

# 커스텀 데이터로 파인튜닝
model.train(
    data='ppe_dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='ppe_detector'
)

# 추론
results = model('construction_site.jpg')
results[0].show()
```

#### 추천 사전 학습 가중치
- **YOLOv8n**: 빠른 추론 (모바일/엣지)
- **YOLOv8s**: 균형 (일반적)
- **YOLOv8m**: 정확도 우선
- **YOLOv8l/x**: 최고 정확도

### 2. Detectron2 기반 모델

```python
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
)

predictor = DefaultPredictor(cfg)
outputs = predictor(image)
```

### 3. Pose Estimation 모델

#### MediaPipe Pose
```python
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5
)

results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
landmarks = results.pose_landmarks
```

#### OpenPose
- **링크**: https://github.com/CMU-Perceptual-Computing-Lab/openpose
- **특징**: 다중 인물 포즈 추정

#### AlphaPose
- **링크**: https://github.com/MVIG-SJTU/AlphaPose
- **특징**: 정확도 높음, 실시간 가능

---

## 오픈소스 프로젝트

### 1. 완성된 PPE 탐지 프로젝트

#### PPE-Detection-YOLO-Deep_SORT
- **링크**: https://github.com/AnshulSood11/PPE-Detection-YOLOv3-Deep-SORT
- **기능**: PPE 탐지 + 객체 추적
- **기술**: YOLOv3, DeepSORT
- **특징**: 실시간 비디오 처리

#### Hardhat-Detection
- **링크**: https://github.com/wujixiu/helmet-detection
- **기능**: 안전모 착용 여부 탐지
- **기술**: YOLOv5
- **데이터셋**: 포함됨

#### Safety-Helmet-Wearing-Detection
- **링크**: https://github.com/PeterH0323/Smart_Construction
- **기능**: 통합 안전관리 시스템
- **기술**: YOLOv5, DeepSORT
- **특징**: 웹 인터페이스 포함

### 2. 행동 인식 프로젝트

#### Fall-Detection
- **링크**: https://github.com/uttejkumaro/Fall-Detection-using-OpenCV-and-Python
- **기능**: 낙상 감지
- **기술**: OpenCV, Pose Estimation

#### Action-Recognition
- **링크**: https://github.com/HowieMa/lstm_pm_pytorch
- **기능**: 작업 행동 인식
- **기술**: LSTM, Pose

### 3. 통합 시스템

#### Construction-Site-Monitoring
- **링크**: https://github.com/SamSamhuns/yolov5_onnx
- **기능**: 실시간 모니터링 시스템
- **기술**: YOLOv5, ONNX, FastAPI

---

## 연구 논문

### 1. PPE 탐지 관련

#### "Automatic Detection of Hardhats Worn by Construction Personnel: A Deep Learning Approach"
- **저자**: Wu et al.
- **연도**: 2019
- **학회**: ASCE International Conference on Computing in Civil Engineering
- **요약**: CNN 기반 안전모 탐지
- **링크**: [ResearchGate](https://www.researchgate.net/)

#### "Deep Learning-Based Object Detection in Construction Sites"
- **저자**: Fang et al.
- **연도**: 2020
- **저널**: Automation in Construction
- **주요 내용**: YOLOv3를 활용한 건설 현장 객체 탐지

#### "Personal Protective Equipment Detection Using YOLO"
- **저자**: Hayat et al.
- **연도**: 2021
- **주요 내용**: 실시간 PPE 탐지 시스템 설계

### 2. 건설 안전 관리

#### "Computer Vision Applications in Construction Safety: A Review"
- **저자**: Fang et al.
- **연도**: 2020
- **저널**: Automation in Construction
- **요약**: Computer Vision 기술의 건설 안전 적용 사례 리뷰

#### "Automated Vision-Based Recognition of Construction Worker Actions"
- **저자**: Han & Lee
- **연도**: 2013
- **저널**: Automation in Construction
- **주요 내용**: 작업자 행동 자동 인식

### 3. 최신 연구 동향

#### "Vision-Based Safety Management in Construction: A Review"
- **연도**: 2022
- **주요 내용**: 최신 Vision 기술 동향 정리

#### "Deep Learning for Construction Safety"
- **연도**: 2023
- **주요 내용**: 딥러닝의 건설 안전 적용

**논문 검색:**
- [Google Scholar](https://scholar.google.com/) - "construction safety deep learning"
- [arXiv](https://arxiv.org/) - cs.CV 카테고리
- [Papers with Code](https://paperswithcode.com/)

---

## 상용 솔루션

### 1. 국내 솔루션

#### 스마트 안전관리 시스템
- **제공사**: 현대건설, 삼성물산 등 대형 건설사 자체 개발
- **기능**:
  - AI 기반 PPE 탐지
  - 위험 구역 모니터링
  - 실시간 알림
  - 통계 대시보드

#### AI 안전관제 플랫폼
- **제공사**: 건설 ICT 스타트업
- **특징**: 클라우드 기반 SaaS

### 2. 해외 솔루션

#### Smartvid.io
- **링크**: https://www.smartvid.io/
- **기능**: AI 기반 안전 및 품질 관리
- **특징**: 자동 위험 요소 탐지

#### Buildots
- **링크**: https://buildots.com/
- **기능**: 360도 카메라 + AI 분석
- **특징**: 공정 관리 + 안전 관리 통합

#### Vinnie.ai
- **링크**: https://vinnie.ai/
- **기능**: 비디오 기반 안전 모니터링
- **특징**: 자동 규정 준수 확인

### 3. 하드웨어 솔루션

#### AI 카메라
- **AXIS Q1656**: AI 분석 기능 내장
- **Hikvision DeepinMind**: 딥러닝 카메라
- **Dahua WizMind**: AI 기능 카메라

#### 엣지 컴퓨팅 디바이스
- **NVIDIA Jetson Orin**: 고성능 엣지 AI
- **NVIDIA Jetson Xavier NX**: 중급 성능
- **NVIDIA Jetson Nano**: 입문용
- **Intel NUC + Movidius**: Intel 기반

---

## 개발 방법론

### 1. 프로젝트 구조

```
construction_safety_system/
│
├── data/
│   ├── raw/                    # 원본 데이터
│   ├── processed/              # 전처리된 데이터
│   ├── annotations/            # 어노테이션 파일
│   └── datasets/               # 학습/검증/테스트 분할
│
├── models/
│   ├── detection/              # 객체 탐지 모델
│   ├── classification/         # 분류 모델
│   ├── tracking/               # 추적 모델
│   └── checkpoints/            # 학습 체크포인트
│
├── src/
│   ├── data/
│   │   ├── preprocessing.py
│   │   ├── augmentation.py
│   │   └── dataset.py
│   │
│   ├── models/
│   │   ├── detector.py
│   │   ├── classifier.py
│   │   └── tracker.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   ├── validate.py
│   │   └── config.py
│   │
│   ├── inference/
│   │   ├── predict.py
│   │   ├── video_processor.py
│   │   └── stream_handler.py
│   │
│   ├── utils/
│   │   ├── visualization.py
│   │   ├── metrics.py
│   │   └── io.py
│   │
│   └── api/
│       ├── app.py              # FastAPI/Flask 앱
│       ├── routes.py
│       └── schemas.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
│
├── tests/
│   ├── test_detector.py
│   ├── test_api.py
│   └── test_utils.py
│
├── configs/
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── deployment_config.yaml
│
├── scripts/
│   ├── download_dataset.sh
│   ├── train_model.sh
│   └── deploy.sh
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── docs/
│   ├── API.md
│   ├── DEPLOYMENT.md
│   └── USER_GUIDE.md
│
├── requirements.txt
├── setup.py
└── README.md
```

### 2. 개발 워크플로우

#### Phase 1: 데이터 수집 및 준비 (1-2주)
```python
# 1. 데이터 수집
# - 공개 데이터셋 다운로드
# - 자체 촬영 (가능한 경우)
# - 데이터 증강

# 2. 어노테이션
# - Roboflow, CVAT 사용
# - 클래스 정의: helmet, no-helmet, vest, no-vest, person

# 3. 데이터 분할
from sklearn.model_selection import train_test_split

images = glob.glob('data/raw/images/*.jpg')
train, val = train_test_split(images, test_size=0.2, random_state=42)
val, test = train_test_split(val, test_size=0.5, random_state=42)
```

#### Phase 2: 모델 학습 (2-3주)
```python
# training/train.py
import torch
from ultralytics import YOLO

def train_ppe_detector():
    model = YOLO('yolov8n.pt')

    results = model.train(
        data='configs/ppe_dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device='cuda:0',
        workers=8,
        patience=20,
        save=True,
        plots=True,

        # 증강
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1
    )

    return results

if __name__ == '__main__':
    results = train_ppe_detector()
```

#### Phase 3: 모델 평가 및 최적화 (1주)
```python
# evaluation/evaluate.py
from ultralytics import YOLO
import json

def evaluate_model(model_path, test_data):
    model = YOLO(model_path)

    # 테스트 세트 평가
    metrics = model.val(data=test_data)

    results = {
        'mAP50': float(metrics.box.map50),
        'mAP50-95': float(metrics.box.map),
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr)
    }

    # 결과 저장
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    return results

# 모델 최적화
def optimize_model(model_path):
    model = YOLO(model_path)

    # ONNX 변환
    model.export(format='onnx', simplify=True)

    # TensorRT 변환 (NVIDIA GPU)
    model.export(format='engine', half=True, device=0)

    # CoreML 변환 (iOS)
    model.export(format='coreml')
```

#### Phase 4: API 개발 (1주)
```python
# api/app.py
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI(title="Construction Safety API")

model = YOLO('models/best.pt')

@app.post("/detect")
async def detect_ppe(file: UploadFile = File(...)):
    """
    이미지에서 PPE 탐지
    """
    # 이미지 읽기
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 추론
    results = model(image)

    # 결과 파싱
    detections = []
    for box in results[0].boxes:
        detections.append({
            'class': model.names[int(box.cls[0])],
            'confidence': float(box.conf[0]),
            'bbox': box.xyxy[0].tolist()
        })

    return {
        'detections': detections,
        'violations': check_violations(detections)
    }

def check_violations(detections):
    """
    안전 규정 위반 확인
    """
    violations = []

    # PPE 착용 여부 확인
    persons = [d for d in detections if d['class'] == 'person']
    helmets = [d for d in detections if d['class'] == 'helmet']

    if len(persons) > len(helmets):
        violations.append({
            'type': 'MISSING_HELMET',
            'count': len(persons) - len(helmets)
        })

    return violations

# 실행: uvicorn api.app:app --reload
```

#### Phase 5: 배포 (1-2주)
```dockerfile
# docker/Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker/docker-compose.yml
version: '3.8'

services:
  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ../models:/app/models
      - ../data:/app/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - CUDA_VISIBLE_DEVICES=0

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: safety_db
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

### 3. 성능 최적화 전략

#### 모델 경량화
```python
# 1. 프루닝 (Pruning)
import torch.nn.utils.prune as prune

def prune_model(model, amount=0.3):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    return model

# 2. 양자화 (Quantization)
from torch.quantization import quantize_dynamic

quantized_model = quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# 3. 지식 증류 (Knowledge Distillation)
# 큰 모델(teacher)의 지식을 작은 모델(student)로 전달
```

#### 추론 최적화
```python
# TensorRT 사용
from torch2trt import torch2trt

model_trt = torch2trt(
    model,
    [x],
    fp16_mode=True,
    max_batch_size=16
)

# ONNX Runtime
import onnxruntime as ort

session = ort.InferenceSession(
    'model.onnx',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

# 배치 처리
def batch_inference(images, batch_size=16):
    results = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        batch_results = model(batch)
        results.extend(batch_results)
    return results
```

### 4. 모니터링 및 로깅

```python
# logging_config.py
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    logger = logging.getLogger('safety_system')
    logger.setLevel(logging.INFO)

    # 파일 핸들러
    file_handler = RotatingFileHandler(
        'logs/system.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)

    # 포맷터
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger

# 사용
logger = setup_logging()
logger.info('System started')
logger.warning('High violation rate detected')
logger.error('Camera connection failed')
```

---

## 규제 및 표준

### 1. 국내 법규

#### 산업안전보건법
- **주관**: 고용노동부
- **주요 내용**:
  - 안전보건관리체계 구축
  - 유해·위험 방지 조치
  - 개인보호구 지급 및 착용
  - 안전보건교육

#### 건설기술진흥법
- **주관**: 국토교통부
- **주요 내용**:
  - 건설공사 안전관리
  - 안전관리계획서 작성

### 2. 국제 표준

#### OSHA (Occupational Safety and Health Administration)
- **국가**: 미국
- **주요 규정**:
  - 29 CFR 1926: 건설 산업 안전
  - PPE 착용 의무화
  - 낙하 방지 조치

#### ISO 45001
- **주제**: 안전보건경영시스템
- **내용**: 국제 표준 안전관리 프레임워크

### 3. PPE 착용 기준

| 작업 유형 | 필수 PPE |
|----------|---------|
| 일반 건설 작업 | 안전모, 안전화 |
| 고소 작업 (2m 이상) | 안전모, 안전화, 안전벨트 |
| 중장비 작업 | 안전모, 안전조끼, 안전화 |
| 용접 작업 | 안전모, 보안경, 용접 장갑 |
| 페인트 작업 | 안전모, 방진 마스크, 보안경 |
| 철골 작업 | 안전모, 안전벨트, 안전화 |

---

## 실전 적용 팁

### 1. 데이터 수집 전략

#### 다양성 확보
- 다양한 날씨 조건 (맑음, 흐림, 비)
- 다양한 시간대 (아침, 점심, 저녁)
- 다양한 각도 및 거리
- 다양한 조명 조건

#### 클래스 불균형 해결
```python
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

# 클래스 가중치 계산
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(labels),
    y=labels
)

# 손실 함수에 가중치 적용
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
```

### 2. 모델 선택 가이드

| 요구사항 | 추천 모델 | 이유 |
|---------|----------|------|
| 실시간 처리 (30fps+) | YOLOv8n, YOLOv7-tiny | 속도 우선 |
| 높은 정확도 | YOLOv8x, Faster R-CNN | 정확도 우선 |
| 엣지 디바이스 | MobileNet, EfficientDet | 경량화 |
| 세그멘테이션 | Mask R-CNN, YOLOv8-seg | 픽셀 단위 분석 |
| 포즈 추정 | MediaPipe, OpenPose | 실시간 가능 |

### 3. 하이퍼파라미터 튜닝

```python
# Optuna를 사용한 자동 튜닝
import optuna

def objective(trial):
    # 하이퍼파라미터 샘플링
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])

    # 모델 학습
    model = train_model(lr, batch_size, optimizer_name)

    # 검증 성능
    val_map = validate(model)

    return val_map

# 최적화 실행
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f'Best params: {study.best_params}')
```

### 4. 에러 처리

```python
class SafetyMonitoringSystem:
    def __init__(self):
        self.error_count = 0
        self.max_errors = 10

    def process_frame(self, frame):
        try:
            results = self.detector.detect(frame)
            self.error_count = 0  # 성공 시 에러 카운트 리셋
            return results

        except Exception as e:
            self.error_count += 1
            logger.error(f'Detection error: {e}')

            if self.error_count >= self.max_errors:
                logger.critical('Too many errors, restarting system')
                self.restart()

            return None

    def restart(self):
        """시스템 재시작"""
        logger.info('Restarting detection system...')
        self.detector = SafetyDetector('models/best.pt')
        self.error_count = 0
```

---

## 추가 리소스

### 커뮤니티
- [Reddit - r/computervision](https://www.reddit.com/r/computervision/)
- [Papers with Code](https://paperswithcode.com/)
- [Roboflow Community](https://discuss.roboflow.com/)

### 블로그 및 튜토리얼
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [PyImageSearch](https://www.pyimagesearch.com/)
- [Towards Data Science](https://towardsdatascience.com/)

### YouTube 채널
- [Nicolai Nielsen](https://www.youtube.com/@NicolaiAI)
- [Augmented Startups](https://www.youtube.com/@AugmentedStartups)
- [Computer Vision Engineer](https://www.youtube.com/@ComputerVisionEngineer)

---

**이 문서는 지속적으로 업데이트됩니다. 최신 정보는 공식 문서를 참고하세요.**
