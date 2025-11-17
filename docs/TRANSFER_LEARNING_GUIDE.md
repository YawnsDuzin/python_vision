# Transfer Learning vs Training from Scratch
# 전이 학습 vs 처음부터 학습

Vision 프로젝트에서 모델을 선택하고 학습하는 전략 가이드입니다.

---

## TL;DR (결론부터)

**✅ 99%의 경우, 사전 학습된 모델을 파인튜닝하는 것이 정답입니다.**

### 이유:
1. **데이터 부족**: 처음부터 학습하려면 수백만 장의 이미지가 필요
2. **학습 시간**: 사전 학습 모델은 수일~수주 걸린 학습 결과물
3. **성능**: 전이 학습이 대부분 더 좋은 성능
4. **비용**: GPU 학습 비용이 수백만원 절약
5. **실무 표준**: 구글, 메타 등 빅테크도 이 방식 사용

---

## 1. 실무에서의 일반적인 접근법

### 가장 흔한 시나리오 (95%)

```python
from ultralytics import YOLO

# ✅ 추천: 사전 학습된 모델 로드
model = YOLO('yolov8n.pt')  # COCO 데이터셋으로 사전 학습됨

# 커스텀 데이터로 파인튜닝
model.train(
    data='ppe_dataset.yaml',
    epochs=100,
    imgsz=640,
    pretrained=True  # 사전 학습 가중치 사용
)
```

### 거의 사용하지 않는 방법 (5%)

```python
# ❌ 비추천: 처음부터 학습
model = YOLO('yolov8n.yaml')  # 아키텍처만 로드 (가중치 없음)

model.train(
    data='ppe_dataset.yaml',
    epochs=300,  # 훨씬 더 많은 에폭 필요
    imgsz=640,
    pretrained=False  # 랜덤 초기화
)
```

---

## 2. 전이 학습(Transfer Learning)이란?

### 개념
대규모 데이터셋(ImageNet, COCO 등)으로 사전 학습된 모델을 가져와서,
내 작업(Task)에 맞게 **일부만 재학습**하는 방법

### 왜 효과적인가?
```
사전 학습된 모델이 학습한 것:
├── 저수준 특징 (Low-level features)
│   ├── 엣지 (edges)
│   ├── 코너 (corners)
│   ├── 색상 패턴 (color patterns)
│   └── 텍스처 (textures)
│
└── 고수준 특징 (High-level features)
    ├── 형태 (shapes)
    ├── 객체 부분 (object parts)
    └── 객체 개념 (object concepts)

👉 이런 특징들은 모든 이미지 작업에서 공통적으로 사용됨!
```

---

## 3. 사전 학습 모델 선택 가이드

### 주요 사전 학습 데이터셋

#### ImageNet
- **규모**: 1,400만 이미지, 1,000 클래스
- **용도**: 이미지 분류
- **특징**: 일반적인 객체 인식
- **사용 모델**: ResNet, VGG, EfficientNet, MobileNet

```python
import torchvision.models as models

# ImageNet으로 사전 학습된 ResNet
model = models.resnet50(pretrained=True)
```

#### COCO (Common Objects in Context)
- **규모**: 33만 이미지, 80 클래스
- **용도**: 객체 탐지, 세그멘테이션
- **특징**: 복잡한 씬, 여러 객체
- **사용 모델**: YOLO, Faster R-CNN, Mask R-CNN

```python
from ultralytics import YOLO

# COCO로 사전 학습된 YOLOv8
model = YOLO('yolov8n.pt')  # COCO 80 classes
```

#### OpenImages
- **규모**: 900만 이미지, 600 클래스
- **용도**: 객체 탐지
- **특징**: 더 다양한 클래스

---

## 4. 실전 전략: 3단계 접근법

### 📊 데이터셋 크기에 따른 전략

```
데이터 양          전략                        파인튜닝 범위
─────────────────────────────────────────────────────────
< 100 이미지       데이터 증강 + 전이 학습        마지막 레이어만
100 ~ 1,000       전이 학습                     마지막 몇 개 레이어
1,000 ~ 10,000    전이 학습                     상위 절반
10,000 ~ 100,000  전이 학습                     전체 파인튜닝
> 100,000         전이 학습 or 처음부터          전체 학습 가능
```

### 전략 1: 극소량 데이터 (< 100장)

**권장 사항**: 데이터 증강 + 마지막 레이어만 학습

```python
import torch
import torch.nn as nn
from torchvision import models, transforms

# 1. 사전 학습 모델 로드
model = models.resnet50(pretrained=True)

# 2. 모든 레이어 동결
for param in model.parameters():
    param.requires_grad = False

# 3. 마지막 레이어만 교체 (안전모 착용/미착용 2클래스)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# 4. 마지막 레이어만 학습 가능
for param in model.fc.parameters():
    param.requires_grad = True

# 5. 강력한 데이터 증강
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 6. 학습 (마지막 레이어만)
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
```

### 전략 2: 소량 데이터 (100~1,000장)

**권장 사항**: 상위 레이어만 파인튜닝

```python
model = models.resnet50(pretrained=True)

# 하위 레이어 동결 (특징 추출기)
for name, param in model.named_parameters():
    if 'layer4' not in name and 'fc' not in name:
        param.requires_grad = False

# 상위 레이어만 학습
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# 차등 학습률 (Discriminative Learning Rates)
optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
])
```

### 전략 3: 중간 데이터 (1,000~10,000장)

**권장 사항**: 전체 네트워크 파인튜닝 (낮은 학습률)

```python
model = models.resnet50(pretrained=True)

# 마지막 레이어 교체
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 5)  # PPE 5 클래스

# 전체 네트워크 학습 (차등 학습률)
optimizer = torch.optim.Adam([
    {'params': model.layer1.parameters(), 'lr': 1e-5},
    {'params': model.layer2.parameters(), 'lr': 1e-5},
    {'params': model.layer3.parameters(), 'lr': 1e-4},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
])
```

### 전략 4: 대량 데이터 (> 10,000장)

**권장 사항**: 전체 파인튜닝 또는 처음부터 학습 고려

```python
# 방법 1: 전체 파인튜닝 (여전히 추천)
model = YOLO('yolov8m.pt')  # 중형 모델
model.train(
    data='large_dataset.yaml',
    epochs=200,
    imgsz=640,
    batch=32,
    pretrained=True
)

# 방법 2: 처음부터 학습 (선택사항)
model = YOLO('yolov8m.yaml')  # 아키텍처만
model.train(
    data='large_dataset.yaml',
    epochs=500,  # 더 많은 에폭
    imgsz=640,
    batch=32,
    pretrained=False
)
```

---

## 5. YOLO 전이 학습 실전 예시

### YOLOv8 파인튜닝 (공사현장 안전)

```python
from ultralytics import YOLO

# 1. 모델 선택 (크기별)
model_options = {
    'nano': 'yolov8n.pt',      # 빠름, 모바일
    'small': 'yolov8s.pt',     # 균형
    'medium': 'yolov8m.pt',    # 정확도 우선
    'large': 'yolov8l.pt',     # 최고 성능
    'xlarge': 'yolov8x.pt'     # 최고 정확도
}

# 2. 사전 학습 모델 로드
model = YOLO('yolov8s.pt')  # COCO로 사전 학습됨

# 3. 데이터셋 준비
# ppe_dataset.yaml:
"""
path: /path/to/dataset
train: train/images
val: val/images

nc: 5
names: ['person', 'helmet', 'no-helmet', 'vest', 'no-vest']
"""

# 4. 파인튜닝
results = model.train(
    # 데이터
    data='ppe_dataset.yaml',

    # 학습 설정
    epochs=100,
    batch=16,
    imgsz=640,

    # 중요: pretrained=True (기본값)
    pretrained=True,

    # 최적화
    optimizer='Adam',
    lr0=0.001,  # 초기 학습률 (낮게 설정)
    lrf=0.01,   # 최종 학습률

    # 정규화
    weight_decay=0.0005,
    dropout=0.0,

    # 데이터 증강
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10,
    translate=0.1,
    scale=0.5,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,

    # 기타
    patience=20,  # Early stopping
    save=True,
    plots=True,
    device=0  # GPU
)

# 5. 평가
metrics = model.val()
print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")
```

### 학습 곡선 예시

```
Epoch   GPU_mem   box_loss   cls_loss   dfl_loss   Precision   Recall   mAP50   mAP50-95
────────────────────────────────────────────────────────────────────────────────────────
1/100   4.2G      1.234      2.345      1.456      0.523       0.456    0.487   0.312
10/100  4.2G      0.845      1.234      1.123      0.734       0.678    0.701   0.523
50/100  4.2G      0.523      0.678      0.845      0.856       0.823    0.834   0.687
100/100 4.2G      0.456      0.589      0.734      0.889       0.867    0.876   0.745
                                                                        ↑
                                                                    목표: > 0.8
```

---

## 6. 레이어 동결(Freezing) 전략

### 시각적 이해

```
┌─────────────────────────────────────┐
│         YOLOv8 Architecture         │
├─────────────────────────────────────┤
│ Backbone (CSPDarknet)               │
│  ├─ Conv Layers  [FROZEN] ❄️        │  ← 일반적 특징
│  ├─ C2f Blocks   [FROZEN] ❄️        │  ← 엣지, 텍스처
│  └─ SPPF         [FROZEN] ❄️        │
├─────────────────────────────────────┤
│ Neck (PANet)                        │
│  ├─ Upsample     [TRAINABLE] 🔥     │  ← 특화 특징
│  └─ Concat       [TRAINABLE] 🔥     │
├─────────────────────────────────────┤
│ Head (Detection)                    │
│  ├─ Conv         [TRAINABLE] 🔥     │  ← 클래스별
│  └─ Detect       [TRAINABLE] 🔥     │  ← 작업 특화
└─────────────────────────────────────┘
```

### 코드 구현

```python
import torch

# 모델 로드
model = YOLO('yolov8s.pt')

# 방법 1: YOLOv8 내장 freeze 기능
model.train(
    data='ppe_dataset.yaml',
    epochs=100,
    freeze=10  # 처음 10개 레이어 동결
)

# 방법 2: PyTorch로 직접 제어
for i, (name, param) in enumerate(model.model.named_parameters()):
    if i < 10:  # 처음 10개 레이어
        param.requires_grad = False
        print(f"Frozen: {name}")
    else:
        param.requires_grad = True
        print(f"Trainable: {name}")
```

---

## 7. 처음부터 학습해야 하는 경우

### ❌ 대부분은 필요 없지만, 다음 경우는 고려 가능:

1. **완전히 다른 도메인**
   - 의료 영상 (X-ray, CT, MRI)
   - 위성 사진 (RGB 아님, 다중 스펙트럼)
   - 특수 센서 데이터 (적외선, 레이더)

2. **매우 특수한 아키텍처**
   - 기존 모델로는 불가능한 작업
   - 논문 연구 목적

3. **충분한 데이터 + 리소스**
   - 100만 장 이상 데이터
   - GPU 클러스터 사용 가능
   - 수주간 학습 가능

### 예시: 의료 영상

```python
# X-ray 이미지는 일반 RGB와 매우 다름
# 하지만 여전히 ImageNet 사전 학습 모델이 도움됨!

model = models.resnet50(pretrained=True)

# 첫 번째 Conv 레이어를 1채널로 수정
model.conv1 = nn.Conv2d(
    1,  # 그레이스케일
    64,
    kernel_size=7,
    stride=2,
    padding=3,
    bias=False
)

# 사전 학습 가중치의 평균을 1채널에 적용
original_weights = models.resnet50(pretrained=True).conv1.weight.data
model.conv1.weight.data = original_weights.mean(dim=1, keepdim=True)

# 전체 파인튜닝
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

---

## 8. 실무 체크리스트

### ✅ 프로젝트 시작 전 확인사항

```markdown
□ 데이터셋 크기 확인
  - < 1,000장: 전이 학습 필수
  - > 10,000장: 전이 학습 권장
  - > 100,000장: 전이 학습 또는 처음부터

□ 유사한 사전 학습 모델 찾기
  - 객체 탐지: YOLO, Faster R-CNN (COCO)
  - 이미지 분류: ResNet, EfficientNet (ImageNet)
  - 세그멘테이션: Mask R-CNN, U-Net

□ 도메인 유사도 평가
  - 유사함 (일반 객체): 상위 레이어만 파인튜닝
  - 다름 (특수 도메인): 전체 파인튜닝

□ 컴퓨팅 리소스 확인
  - GPU 메모리
  - 학습 시간 예산
```

### 🎯 성능 개선 팁

1. **학습률 조정**
```python
# 좋은 시작점
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 성능이 안 나오면
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.1,
    patience=5
)
```

2. **데이터 증강 활용**
```python
from albumentations import (
    Compose, RandomBrightnessContrast, RandomRotate90,
    HorizontalFlip, ShiftScaleRotate, Blur
)

transform = Compose([
    RandomRotate90(p=0.5),
    HorizontalFlip(p=0.5),
    RandomBrightnessContrast(p=0.3),
    ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
    Blur(blur_limit=3, p=0.3)
])
```

3. **앙상블 활용**
```python
# 여러 사전 학습 모델 앙상블
models = [
    YOLO('yolov8s.pt'),
    YOLO('yolov8m.pt'),
    YOLO('yolov8l.pt')
]

# 각각 파인튜닝 후 예측 결합
predictions = []
for model in models:
    pred = model(image)
    predictions.append(pred)

# 투표 또는 평균
final_prediction = ensemble(predictions)
```

---

## 9. 비용 및 시간 비교

### 실제 프로젝트 예시 (PPE 탐지, 5,000장)

| 항목 | 전이 학습 | 처음부터 학습 |
|------|----------|--------------|
| **데이터 필요량** | 5,000장 ✅ | 100,000장+ ❌ |
| **학습 시간** | 2-4시간 | 2-7일 |
| **GPU 비용** | $5-10 | $100-500 |
| **mAP50 성능** | 0.85-0.90 | 0.70-0.80 (데이터 부족) |
| **수렴 속도** | 50-100 epoch | 300-500 epoch |
| **난이도** | 쉬움 | 어려움 |

### 💰 AWS GPU 비용 (p3.2xlarge, V100 1개)

```
전이 학습:
- 학습 시간: 4시간
- 비용: 4시간 × $3.06/시간 = $12.24

처음부터 학습:
- 학습 시간: 72시간
- 비용: 72시간 × $3.06/시간 = $220.32

절약: $208.08 (95% 절감)
```

---

## 10. 결론 및 권장사항

### 🎓 학습 단계별 권장

**초급 (학습 중)**
- ✅ 무조건 전이 학습 사용
- ✅ 마지막 레이어만 수정
- ✅ 소량 데이터로 실험

**중급 (프로젝트 진행)**
- ✅ 전이 학습 + 전체 파인튜닝
- ✅ 데이터 증강 적극 활용
- ✅ 하이퍼파라미터 튜닝

**고급 (연구/프로덕션)**
- ✅ 전이 학습 기본
- ⚠️ 필요시에만 처음부터 학습 고려
- ✅ 앙상블, 지식 증류 등 고급 기법

### 📝 핵심 요약

```python
# ✅ 이렇게 하세요 (99% 케이스)
model = YOLO('yolov8s.pt')  # 사전 학습 모델
model.train(data='my_data.yaml', epochs=100)

# ❌ 이렇게 하지 마세요 (특수한 경우 제외)
model = YOLO('yolov8s.yaml')  # 빈 모델
model.train(data='my_data.yaml', epochs=500)
```

### 🚀 실무 조언

1. **항상 사전 학습 모델부터 시작**
2. **작게 시작해서 점진적으로 확장**
3. **성능이 안 나올 때 체크리스트**:
   - 데이터 품질 확인
   - 데이터 증강 추가
   - 학습률 조정
   - 더 큰 사전 학습 모델 시도
   - 마지막으로만 처음부터 학습 고려

---

**Remember**: 구글, OpenAI, Meta 같은 빅테크도 사전 학습 모델을 기반으로 합니다.
처음부터 학습하는 것은 예외적인 경우이지, 표준이 아닙니다!
