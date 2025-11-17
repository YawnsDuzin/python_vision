# 멀티태스크 Vision 시스템 구축 가이드
# 이미지 분류 + 객체 탐지 통합

이미지 분류와 객체 탐지를 함께 사용하는 다양한 방법과 실전 예시를 정리합니다.

---

## 목차
1. [접근 방법 개요](#1-접근-방법-개요)
2. [방법 1: 객체 탐지만 사용 (가장 간단)](#2-방법-1-객체-탐지만-사용)
3. [방법 2: 순차적 처리 (두 모델 연결)](#3-방법-2-순차적-처리)
4. [방법 3: 멀티태스크 학습 (하나의 모델)](#4-방법-3-멀티태스크-학습)
5. [방법 4: 앙상블 (여러 모델 조합)](#5-방법-4-앙상블)
6. [실전 예시: 공사현장 안전관리](#6-실전-예시-공사현장-안전관리)

---

## 1. 접근 방법 개요

### 핵심 포인트: 객체 탐지는 이미 분류를 포함합니다!

```
객체 탐지 모델의 출력:
┌─────────────────────────────────────┐
│ Detection 1:                        │
│  - bbox: [100, 200, 150, 250]       │
│  - class: "person" ← 분류!          │
│  - confidence: 0.95                 │
├─────────────────────────────────────┤
│ Detection 2:                        │
│  - bbox: [120, 180, 140, 210]       │
│  - class: "helmet" ← 분류!          │
│  - confidence: 0.89                 │
└─────────────────────────────────────┘

→ 객체 탐지 = 위치 찾기 + 클래스 분류
```

### 언제 두 가지를 함께 사용하나?

| 시나리오 | 필요한 작업 | 추천 방법 |
|---------|-----------|----------|
| **객체별 클래스만 필요** | 객체 탐지만 | 방법 1 |
| **전체 이미지 분류 + 객체 탐지** | 둘 다 | 방법 2, 3 |
| **객체 탐지 + 세부 분류** | 객체 탐지 → 분류 | 방법 2 |
| **여러 관점의 분석** | 여러 모델 | 방법 4 |

### 4가지 접근 방법

```
방법 1: 객체 탐지만 사용 (추천 ★★★★★)
  - YOLO 하나로 끝
  - 위치 + 클래스 동시 출력

방법 2: 순차적 처리 (★★★★☆)
  - 전체 이미지 분류 → 객체 탐지
  - 또는: 객체 탐지 → 각 객체 재분류

방법 3: 멀티태스크 학습 (★★★☆☆)
  - 하나의 모델, 여러 헤드
  - 전체 분류 + 객체 탐지 동시

방법 4: 앙상블 (★★☆☆☆)
  - 여러 모델 결과 종합
  - 높은 정확도, 느린 속도
```

---

## 2. 방법 1: 객체 탐지만 사용 (가장 간단)

### ✅ 대부분의 경우 이것만으로 충분!

**이유**: 객체 탐지 모델은 이미 각 객체를 분류합니다.

#### YOLOv8 예시

```python
from ultralytics import YOLO
import cv2

# 모델 로드 (COCO 80 클래스)
model = YOLO('yolov8s.pt')

# 이미지 추론
results = model('construction_site.jpg')

# 결과 확인
for result in results:
    boxes = result.boxes

    for box in boxes:
        # 객체 위치
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        # 객체 클래스 (분류!)
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        # 신뢰도
        confidence = float(box.conf[0])

        print(f"Found {class_name} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}] "
              f"with confidence {confidence:.2f}")

"""
출력:
Found person at [100, 200, 150, 300] with confidence 0.95
Found helmet at [120, 180, 140, 210] with confidence 0.89
Found vest at [105, 250, 145, 280] with confidence 0.87
"""
```

#### 커스텀 클래스로 학습

```python
# PPE 탐지 (5 클래스)
"""
ppe_dataset.yaml:
nc: 5
names: ['person', 'helmet', 'no-helmet', 'vest', 'no-vest']
"""

model = YOLO('yolov8s.pt')
model.train(data='ppe_dataset.yaml', epochs=100)

# 추론
results = model('worker.jpg')

# 결과: 각 객체의 위치 + 클래스
for box in results[0].boxes:
    print(f"{model.names[int(box.cls)]}: {box.conf[0]:.2f}")

"""
person: 0.95
helmet: 0.89
vest: 0.87
"""
```

### 이 방법의 한계

```python
# 객체 탐지로 알 수 있는 것:
✅ 사람이 어디 있나? → bbox
✅ 안전모를 쓰고 있나? → helmet 객체 탐지
✅ 안전조끼를 입고 있나? → vest 객체 탐지

# 객체 탐지로 알 수 없는 것:
❌ 전체 이미지가 어떤 씬인가? (실내/실외/공사장/사무실)
❌ 작업 유형은? (용접/페인트/철골)
❌ 날씨는? (맑음/흐림/비)
```

**→ 전체 이미지 분류가 필요하면 방법 2, 3 사용!**

---

## 3. 방법 2: 순차적 처리 (두 모델 연결)

### 패턴 A: 전체 분류 → 객체 탐지

**사용 사례**: 먼저 씬을 파악하고, 적합한 탐지 수행

```python
from transformers import pipeline
from ultralytics import YOLO
import cv2

# ===== 1단계: 전체 이미지 분류 =====
scene_classifier = pipeline(
    "image-classification",
    model="microsoft/resnet-50"
)

image_path = "site.jpg"
scene_result = scene_classifier(image_path)
scene = scene_result[0]['label']

print(f"Scene detected: {scene}")  # "construction site"

# ===== 2단계: 씬에 맞는 객체 탐지 =====
if 'construction' in scene.lower():
    # 공사장용 탐지 모델
    detector = YOLO('models/construction_ppe.pt')

elif 'office' in scene.lower():
    # 사무실용 탐지 모델
    detector = YOLO('models/office_items.pt')

else:
    # 일반 탐지 모델
    detector = YOLO('yolov8s.pt')

# 객체 탐지 수행
detections = detector(image_path)

# ===== 통합 결과 =====
result = {
    'scene': scene,
    'objects': []
}

for box in detections[0].boxes:
    result['objects'].append({
        'class': detector.names[int(box.cls)],
        'confidence': float(box.conf),
        'bbox': box.xyxy[0].tolist()
    })

print(result)
"""
{
    'scene': 'construction site',
    'objects': [
        {'class': 'person', 'confidence': 0.95, 'bbox': [100, 200, 150, 300]},
        {'class': 'helmet', 'confidence': 0.89, 'bbox': [120, 180, 140, 210]}
    ]
}
"""
```

### 패턴 B: 객체 탐지 → 각 객체 재분류

**사용 사례**: 객체를 찾은 후 세부 속성 분류

```python
from ultralytics import YOLO
from transformers import ViTImageProcessor, ViTForImageClassification
import torch
from PIL import Image
import cv2

# ===== 1단계: 객체 탐지 (사람 찾기) =====
detector = YOLO('yolov8s.pt')
image = cv2.imread('workers.jpg')
detections = detector(image)

# ===== 2단계: 각 사람별 PPE 분류 =====
# 안전모 분류기
helmet_processor = ViTImageProcessor.from_pretrained("helmet-classifier")
helmet_model = ViTForImageClassification.from_pretrained("helmet-classifier")

# 안전조끼 분류기
vest_processor = ViTImageProcessor.from_pretrained("vest-classifier")
vest_model = ViTForImageClassification.from_pretrained("vest-classifier")

results = []

for box in detections[0].boxes:
    class_name = detector.names[int(box.cls)]

    if class_name == 'person':
        # 바운딩 박스 추출
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        person_crop = image[y1:y2, x1:x2]
        person_pil = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))

        # 안전모 분류
        helmet_inputs = helmet_processor(person_pil, return_tensors="pt")
        with torch.no_grad():
            helmet_outputs = helmet_model(**helmet_inputs)
            helmet_pred = helmet_outputs.logits.argmax(-1).item()
            has_helmet = helmet_model.config.id2label[helmet_pred]

        # 안전조끼 분류
        vest_inputs = vest_processor(person_pil, return_tensors="pt")
        with torch.no_grad():
            vest_outputs = vest_model(**vest_inputs)
            vest_pred = vest_outputs.logits.argmax(-1).item()
            has_vest = vest_model.config.id2label[vest_pred]

        results.append({
            'bbox': [x1, y1, x2, y2],
            'helmet': has_helmet,  # 'wearing' or 'not_wearing'
            'vest': has_vest,      # 'wearing' or 'not_wearing'
            'violation': has_helmet == 'not_wearing' or has_vest == 'not_wearing'
        })

# ===== 결과 =====
print(f"Found {len(results)} workers")
violations = [r for r in results if r['violation']]
print(f"Safety violations: {len(violations)}")

for i, worker in enumerate(results):
    print(f"Worker {i+1}: Helmet={worker['helmet']}, Vest={worker['vest']}")

"""
Found 3 workers
Safety violations: 1
Worker 1: Helmet=wearing, Vest=wearing
Worker 2: Helmet=not_wearing, Vest=wearing  ← 위반!
Worker 3: Helmet=wearing, Vest=wearing
"""
```

### 실용적인 통합 클래스

```python
class MultiTaskSafetyDetector:
    """
    전체 씬 분류 + 객체 탐지 + 세부 분류
    """
    def __init__(self):
        # 씬 분류기
        self.scene_classifier = pipeline(
            "image-classification",
            model="scene-classifier"
        )

        # 객체 탐지기
        self.object_detector = YOLO('yolov8s.pt')

        # 세부 분류기들
        self.helmet_classifier = ViTForImageClassification.from_pretrained(
            "helmet-classifier"
        )
        self.vest_classifier = ViTForImageClassification.from_pretrained(
            "vest-classifier"
        )

    def analyze(self, image_path):
        """
        통합 분석
        """
        # 1. 씬 분류
        scene = self.scene_classifier(image_path)[0]['label']

        # 2. 객체 탐지
        image = cv2.imread(image_path)
        detections = self.object_detector(image)

        # 3. 각 사람별 세부 분류
        workers = []
        for box in detections[0].boxes:
            if self.object_detector.names[int(box.cls)] == 'person':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_crop = image[y1:y2, x1:x2]

                # 세부 분류
                helmet_status = self._classify_helmet(person_crop)
                vest_status = self._classify_vest(person_crop)

                workers.append({
                    'bbox': [x1, y1, x2, y2],
                    'helmet': helmet_status,
                    'vest': vest_status
                })

        return {
            'scene': scene,
            'worker_count': len(workers),
            'workers': workers,
            'violations': self._check_violations(workers)
        }

    def _classify_helmet(self, image_crop):
        # 헬멧 분류 로직
        ...

    def _classify_vest(self, image_crop):
        # 조끼 분류 로직
        ...

    def _check_violations(self, workers):
        violations = []
        for i, worker in enumerate(workers):
            if worker['helmet'] == 'not_wearing':
                violations.append(f"Worker {i+1}: No helmet")
            if worker['vest'] == 'not_wearing':
                violations.append(f"Worker {i+1}: No vest")
        return violations


# 사용
detector = MultiTaskSafetyDetector()
result = detector.analyze('construction_site.jpg')

print(f"Scene: {result['scene']}")
print(f"Workers: {result['worker_count']}")
print(f"Violations: {len(result['violations'])}")
for violation in result['violations']:
    print(f"  - {violation}")
```

---

## 4. 방법 3: 멀티태스크 학습 (하나의 모델)

### 개념: 하나의 백본, 여러 헤드

```
Input Image
    ↓
┌─────────────────────┐
│   Shared Backbone   │  ResNet, ViT 등
│   (특징 추출)        │  ← 한 번만 계산!
└─────────────────────┘
    ↓
    ├─────────────┬─────────────┬─────────────┐
    ↓             ↓             ↓             ↓
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│Scene    │ │Detection│ │Helmet   │ │Vest     │
│Head     │ │Head     │ │Head     │ │Head     │
└─────────┘ └─────────┘ └─────────┘ └─────────┘
    ↓             ↓             ↓             ↓
  씬분류       객체탐지      헬멧분류      조끼분류
```

### PyTorch 구현

```python
import torch
import torch.nn as nn
from torchvision import models

class MultiTaskSafetyModel(nn.Module):
    """
    멀티태스크 안전관리 모델
    - 씬 분류
    - 객체 탐지
    - PPE 분류
    """
    def __init__(self, num_scene_classes=5, num_object_classes=10):
        super().__init__()

        # 공유 백본 (사전 학습 ResNet)
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # 특징맵 크기: (batch, 2048, H/32, W/32)

        # ===== Head 1: 씬 분류 =====
        self.scene_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_scene_classes)
        )

        # ===== Head 2: 객체 탐지 (간단 버전) =====
        # 실제로는 YOLO/Faster R-CNN 헤드 사용
        self.detection_head = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, num_object_classes + 4, 1)  # classes + bbox
        )

        # ===== Head 3: 헬멧 분류 =====
        self.helmet_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # wearing / not_wearing
        )

        # ===== Head 4: 조끼 분류 =====
        self.vest_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # wearing / not_wearing
        )

    def forward(self, x):
        # 백본 (한 번만 계산!)
        features = self.backbone(x)

        # 각 헤드
        scene_out = self.scene_head(features)
        detection_out = self.detection_head(features)
        helmet_out = self.helmet_head(features)
        vest_out = self.vest_head(features)

        return {
            'scene': scene_out,
            'detection': detection_out,
            'helmet': helmet_out,
            'vest': vest_out
        }

# 모델 생성
model = MultiTaskSafetyModel(num_scene_classes=5, num_object_classes=10)

# 추론
image = torch.randn(1, 3, 640, 640)
outputs = model(image)

print(f"Scene shape: {outputs['scene'].shape}")      # (1, 5)
print(f"Detection shape: {outputs['detection'].shape}")  # (1, 14, H, W)
print(f"Helmet shape: {outputs['helmet'].shape}")    # (1, 2)
print(f"Vest shape: {outputs['vest'].shape}")        # (1, 2)
```

### 멀티태스크 학습 루프

```python
def train_multitask_model(model, dataloader, num_epochs=50):
    """
    멀티태스크 학습
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 각 태스크별 손실 함수
    scene_criterion = nn.CrossEntropyLoss()
    detection_criterion = nn.MSELoss()  # 실제로는 YOLO loss
    helmet_criterion = nn.CrossEntropyLoss()
    vest_criterion = nn.CrossEntropyLoss()

    # 태스크별 가중치 (중요도)
    task_weights = {
        'scene': 0.2,
        'detection': 0.4,
        'helmet': 0.2,
        'vest': 0.2
    }

    for epoch in range(num_epochs):
        for batch in dataloader:
            images = batch['image']
            labels = batch['labels']

            # Forward
            outputs = model(images)

            # 각 태스크별 손실 계산
            loss_scene = scene_criterion(outputs['scene'], labels['scene'])
            loss_detection = detection_criterion(outputs['detection'], labels['detection'])
            loss_helmet = helmet_criterion(outputs['helmet'], labels['helmet'])
            loss_vest = vest_criterion(outputs['vest'], labels['vest'])

            # 가중치 합산
            total_loss = (
                task_weights['scene'] * loss_scene +
                task_weights['detection'] * loss_detection +
                task_weights['helmet'] * loss_helmet +
                task_weights['vest'] * loss_vest
            )

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {total_loss.item():.4f}")
            print(f"  Scene: {loss_scene.item():.4f}")
            print(f"  Detection: {loss_detection.item():.4f}")
            print(f"  Helmet: {loss_helmet.item():.4f}")
            print(f"  Vest: {loss_vest.item():.4f}")
```

### 멀티태스크 학습의 장단점

**장점**:
- ✅ 백본 계산 한 번 → 빠른 추론
- ✅ 특징 공유 → 일반화 성능 향상
- ✅ 메모리 효율적 (백본 하나)

**단점**:
- ❌ 학습 복잡도 높음
- ❌ 태스크 간 간섭 가능
- ❌ 하이퍼파라미터 튜닝 어려움

---

## 5. 방법 4: 앙상블 (여러 모델 조합)

### 개념: 여러 모델의 예측을 종합

```python
class EnsembleSafetySystem:
    """
    여러 모델을 앙상블하여 정확도 향상
    """
    def __init__(self):
        # 여러 객체 탐지 모델
        self.detector1 = YOLO('yolov8s.pt')
        self.detector2 = YOLO('yolov8m.pt')
        self.detector3 = YOLO('yolov8l.pt')

        # 여러 분류 모델
        self.classifier1 = ViTForImageClassification.from_pretrained("vit-base")
        self.classifier2 = ViTForImageClassification.from_pretrained("resnet-50")

    def detect_objects_ensemble(self, image):
        """
        앙상블 객체 탐지
        """
        # 각 모델로 탐지
        results1 = self.detector1(image)
        results2 = self.detector2(image)
        results3 = self.detector3(image)

        # 결과 병합 (NMS - Non-Maximum Suppression)
        all_boxes = []

        for result in [results1, results2, results3]:
            for box in result[0].boxes:
                all_boxes.append({
                    'bbox': box.xyxy[0].tolist(),
                    'class': int(box.cls),
                    'conf': float(box.conf)
                })

        # NMS로 중복 제거
        final_boxes = self.non_max_suppression(all_boxes)

        return final_boxes

    def classify_ensemble(self, image):
        """
        앙상블 분류
        """
        # 각 모델로 분류
        pred1 = self.classifier1(image)
        pred2 = self.classifier2(image)

        # 투표 (Voting)
        # 또는 확률 평균 (Averaging)
        ...

    def non_max_suppression(self, boxes, iou_threshold=0.5):
        """
        중복 박스 제거
        """
        # NMS 구현
        ...
        return filtered_boxes


# 사용
system = EnsembleSafetySystem()
result = system.detect_objects_ensemble('site.jpg')
```

---

## 6. 실전 예시: 공사현장 안전관리

### 종합 시스템 (방법 2 사용)

```python
import cv2
import numpy as np
from ultralytics import YOLO
from transformers import pipeline, ViTImageProcessor, ViTForImageClassification
import torch
from PIL import Image
from datetime import datetime

class ConstructionSafetySystem:
    """
    공사현장 종합 안전관리 시스템

    기능:
    1. 씬 분류 (실내/실외, 작업 유형)
    2. 작업자 탐지
    3. PPE 착용 여부 확인
    4. 위반 사항 리포트
    """

    def __init__(self):
        # 1. 씬 분류기 (전체 이미지)
        self.scene_classifier = pipeline(
            "image-classification",
            model="google/vit-base-patch16-224"
        )

        # 2. 작업자 및 장비 탐지
        self.person_detector = YOLO('yolov8s.pt')

        # 3. PPE 세부 탐지 (커스텀 학습)
        self.ppe_detector = YOLO('models/ppe_detector.pt')

        # 4. 위험 구역 정의
        self.danger_zones = []

    def set_danger_zone(self, polygon):
        """
        위험 구역 설정
        polygon: [(x1,y1), (x2,y2), ...]
        """
        self.danger_zones.append(np.array(polygon, dtype=np.int32))

    def analyze_frame(self, image_path_or_frame):
        """
        프레임 분석 (이미지 또는 비디오 프레임)
        """
        # 이미지 로드
        if isinstance(image_path_or_frame, str):
            image = cv2.imread(image_path_or_frame)
        else:
            image = image_path_or_frame

        # ===== 1. 씬 분류 =====
        scene_result = self.scene_classifier(image)
        scene_type = scene_result[0]['label']
        scene_confidence = scene_result[0]['score']

        # ===== 2. 작업자 탐지 =====
        person_results = self.person_detector(image)
        workers = []

        for box in person_results[0].boxes:
            if self.person_detector.names[int(box.cls)] == 'person':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                workers.append({
                    'id': len(workers),
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(box.conf)
                })

        # ===== 3. PPE 탐지 =====
        ppe_results = self.ppe_detector(image)
        ppe_items = {'helmet': [], 'vest': [], 'gloves': []}

        for box in ppe_results[0].boxes:
            class_name = self.ppe_detector.names[int(box.cls)]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if class_name in ppe_items:
                ppe_items[class_name].append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(box.conf)
                })

        # ===== 4. 작업자별 PPE 매칭 =====
        for worker in workers:
            worker['ppe'] = {
                'helmet': self._has_ppe(worker['bbox'], ppe_items['helmet']),
                'vest': self._has_ppe(worker['bbox'], ppe_items['vest']),
                'gloves': self._has_ppe(worker['bbox'], ppe_items['gloves'])
            }

        # ===== 5. 위험 구역 침입 확인 =====
        for worker in workers:
            worker['in_danger_zone'] = self._is_in_danger_zone(
                worker['bbox'],
                self.danger_zones
            )

        # ===== 6. 위반 사항 정리 =====
        violations = self._check_violations(workers)

        # ===== 결과 반환 =====
        return {
            'timestamp': datetime.now().isoformat(),
            'scene': {
                'type': scene_type,
                'confidence': scene_confidence
            },
            'workers': workers,
            'ppe_summary': {
                'helmet_count': len(ppe_items['helmet']),
                'vest_count': len(ppe_items['vest']),
                'gloves_count': len(ppe_items['gloves'])
            },
            'violations': violations,
            'danger_zone_intrusions': sum(1 for w in workers if w['in_danger_zone'])
        }

    def _has_ppe(self, person_bbox, ppe_list):
        """
        작업자가 PPE를 착용했는지 확인 (IoU 기반)
        """
        px1, py1, px2, py2 = person_bbox

        for ppe in ppe_list:
            ex1, ey1, ex2, ey2 = ppe['bbox']

            # IoU 계산
            iou = self._calculate_iou(
                [px1, py1, px2, py2],
                [ex1, ey1, ex2, ey2]
            )

            if iou > 0.3:  # 30% 이상 겹치면 착용으로 판단
                return True

        return False

    def _calculate_iou(self, box1, box2):
        """
        IoU (Intersection over Union) 계산
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def _is_in_danger_zone(self, bbox, zones):
        """
        위험 구역 내부인지 확인
        """
        # 바운딩 박스 중심점
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2

        for zone in zones:
            if cv2.pointPolygonTest(zone, (cx, cy), False) >= 0:
                return True

        return False

    def _check_violations(self, workers):
        """
        안전 규정 위반 확인
        """
        violations = []

        for worker in workers:
            worker_id = worker['id']

            # PPE 미착용
            if not worker['ppe']['helmet']:
                violations.append({
                    'type': 'NO_HELMET',
                    'worker_id': worker_id,
                    'severity': 'HIGH',
                    'message': f"Worker {worker_id} is not wearing a helmet"
                })

            if not worker['ppe']['vest']:
                violations.append({
                    'type': 'NO_VEST',
                    'worker_id': worker_id,
                    'severity': 'MEDIUM',
                    'message': f"Worker {worker_id} is not wearing a vest"
                })

            # 위험 구역 침입
            if worker['in_danger_zone']:
                violations.append({
                    'type': 'DANGER_ZONE',
                    'worker_id': worker_id,
                    'severity': 'CRITICAL',
                    'message': f"Worker {worker_id} is in a danger zone"
                })

        return violations

    def visualize_results(self, image, analysis_result):
        """
        결과 시각화
        """
        vis_image = image.copy()

        # 위험 구역 그리기
        for zone in self.danger_zones:
            overlay = vis_image.copy()
            cv2.fillPoly(overlay, [zone], (0, 0, 255))
            cv2.addWeighted(overlay, 0.3, vis_image, 0.7, 0, vis_image)
            cv2.polylines(vis_image, [zone], True, (0, 0, 255), 2)

        # 작업자별 박스 그리기
        for worker in analysis_result['workers']:
            x1, y1, x2, y2 = worker['bbox']

            # 위반 여부에 따라 색상 결정
            has_violation = any(
                v['worker_id'] == worker['id']
                for v in analysis_result['violations']
            )

            color = (0, 0, 255) if has_violation else (0, 255, 0)

            # 박스
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

            # PPE 상태 텍스트
            status_text = f"ID:{worker['id']} "
            status_text += "H:" + ("✓" if worker['ppe']['helmet'] else "✗")
            status_text += " V:" + ("✓" if worker['ppe']['vest'] else "✗")

            cv2.putText(
                vis_image,
                status_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        # 위반 통계
        violation_text = f"Violations: {len(analysis_result['violations'])}"
        cv2.putText(
            vis_image,
            violation_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        return vis_image


# ===== 사용 예시 =====
def main():
    # 시스템 초기화
    system = ConstructionSafetySystem()

    # 위험 구역 설정
    danger_zone = [(100, 200), (300, 200), (300, 400), (100, 400)]
    system.set_danger_zone(danger_zone)

    # 이미지 분석
    image_path = 'construction_site.jpg'
    result = system.analyze_frame(image_path)

    # 결과 출력
    print("=== Analysis Result ===")
    print(f"Scene: {result['scene']['type']} ({result['scene']['confidence']:.2%})")
    print(f"Workers detected: {len(result['workers'])}")
    print(f"PPE Summary:")
    print(f"  Helmets: {result['ppe_summary']['helmet_count']}")
    print(f"  Vests: {result['ppe_summary']['vest_count']}")
    print(f"Violations: {len(result['violations'])}")

    for violation in result['violations']:
        print(f"  [{violation['severity']}] {violation['message']}")

    # 시각화
    image = cv2.imread(image_path)
    vis_image = system.visualize_results(image, result)
    cv2.imwrite('result.jpg', vis_image)

    # 비디오 처리
    cap = cv2.VideoCapture('construction.mp4')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 분석
        result = system.analyze_frame(frame)

        # 시각화
        vis_frame = system.visualize_results(frame, result)

        cv2.imshow('Safety Monitor', vis_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
```

---

## 7. 방법별 비교

### 성능 비교

| 방법 | 속도 | 정확도 | 복잡도 | 추천도 |
|------|------|--------|--------|--------|
| **방법 1: 객체 탐지만** | ★★★★★ | ★★★★☆ | ★☆☆☆☆ | ★★★★★ |
| **방법 2: 순차 처리** | ★★★☆☆ | ★★★★★ | ★★★☆☆ | ★★★★☆ |
| **방법 3: 멀티태스크** | ★★★★☆ | ★★★★☆ | ★★★★★ | ★★★☆☆ |
| **방법 4: 앙상블** | ★☆☆☆☆ | ★★★★★ | ★★★★☆ | ★★☆☆☆ |

### 사용 사례별 추천

| 사용 사례 | 추천 방법 | 이유 |
|---------|----------|------|
| **PPE 탐지** | 방법 1 | 객체 탐지만으로 충분 |
| **씬별 다른 처리** | 방법 2 | 씬 분류 후 분기 |
| **세부 속성 분석** | 방법 2 | 탐지 후 재분류 |
| **실시간 처리** | 방법 1, 3 | 속도 중요 |
| **최고 정확도** | 방법 4 | 앙상블 |
| **연구 목적** | 방법 3 | 멀티태스크 학습 |

---

## 8. 핵심 요약

### 일반적인 경우

**✅ 대부분: 객체 탐지 모델 하나면 충분!**

```python
# 이것만으로도 충분:
model = YOLO('yolov8s.pt')
results = model('image.jpg')

# 출력: 각 객체의 위치 + 클래스
for box in results[0].boxes:
    print(f"{model.names[int(box.cls)]}: {box.conf[0]:.2f}")
```

### 추가 기능이 필요한 경우

**전체 이미지 분류 필요 → 방법 2 (순차 처리)**

```python
# 1. 씬 분류
scene = scene_classifier(image)

# 2. 씬에 맞는 탐지
detector = select_detector(scene)
objects = detector(image)
```

**세부 속성 분석 필요 → 방법 2 (탐지 → 재분류)**

```python
# 1. 사람 탐지
persons = detector(image)

# 2. 각 사람 재분류
for person in persons:
    helmet_status = helmet_classifier(crop(person))
    vest_status = vest_classifier(crop(person))
```

### 핵심 포인트

1. **객체 탐지 = 위치 + 분류** (이미 포함됨!)
2. **전체 씬 분류가 필요하면** 별도 모델 추가
3. **세부 속성 분석이 필요하면** 순차 처리
4. **실시간 중요하면** 멀티태스크 고려
5. **정확도 최우선이면** 앙상블 고려

**대부분의 경우 YOLO 하나로 끝입니다!** 🎯
