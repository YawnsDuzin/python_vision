# 사전 학습 모델 구조 및 파인튜닝 가이드

대표적인 사전 학습 모델의 내부 구조와 파인튜닝 시 구조 변경 방법을 상세히 설명합니다.

---

## 목차
1. [이미지 분류 모델 (ResNet, VGG, EfficientNet)](#1-이미지-분류-모델)
2. [객체 탐지 모델 (YOLO, Faster R-CNN)](#2-객체-탐지-모델)
3. [세그멘테이션 모델 (Mask R-CNN, U-Net)](#3-세그멘테이션-모델)
4. [파인튜닝 시 구조 변경 패턴](#4-파인튜닝-시-구조-변경-패턴)

---

## 1. 이미지 분류 모델

### 1.1 ResNet-50 구조

ResNet은 **Residual Connection(잔차 연결)**을 사용하여 깊은 네트워크 학습을 가능하게 한 모델입니다.

#### 전체 구조

```
Input Image (224 × 224 × 3)
    ↓
┌─────────────────────────────────────────┐
│ Conv1: 7×7 conv, 64 filters, stride 2  │  ← 초기 특징 추출
│ BatchNorm + ReLU                        │
│ MaxPool: 3×3, stride 2                  │
└─────────────────────────────────────────┘
    ↓ (56 × 56 × 64)
┌─────────────────────────────────────────┐
│ Layer 1: Residual Blocks × 3            │  ← 저수준 특징
│   - Bottleneck(64, 64, 256) × 3         │     (엣지, 텍스처)
└─────────────────────────────────────────┘
    ↓ (56 × 56 × 256)
┌─────────────────────────────────────────┐
│ Layer 2: Residual Blocks × 4            │  ← 중간 수준 특징
│   - Bottleneck(128, 128, 512) × 4       │     (패턴, 형태)
└─────────────────────────────────────────┘
    ↓ (28 × 28 × 512)
┌─────────────────────────────────────────┐
│ Layer 3: Residual Blocks × 6            │  ← 고수준 특징
│   - Bottleneck(256, 256, 1024) × 6      │     (객체 부분)
└─────────────────────────────────────────┘
    ↓ (14 × 14 × 1024)
┌─────────────────────────────────────────┐
│ Layer 4: Residual Blocks × 3            │  ← 최고수준 특징
│   - Bottleneck(512, 512, 2048) × 3      │     (객체 개념)
└─────────────────────────────────────────┘
    ↓ (7 × 7 × 2048)
┌─────────────────────────────────────────┐
│ Global Average Pooling                  │
└─────────────────────────────────────────┘
    ↓ (2048)
┌─────────────────────────────────────────┐
│ FC Layer: 2048 → 1000                   │  ← ImageNet 1000 클래스
└─────────────────────────────────────────┘
    ↓
Output (1000 classes)
```

#### Residual Block 상세 구조

```
Input
  ↓
  ├─────────────────────────────┐
  │                             │
  │  1×1 Conv (차원 축소)        │
  │  ↓                          │
  │  BatchNorm + ReLU           │
  │  ↓                          │
  │  3×3 Conv (특징 추출)        │
  │  ↓                          │
  │  BatchNorm + ReLU           │
  │  ↓                          │
  │  1×1 Conv (차원 복원)        │
  │  ↓                          │
  │  BatchNorm                  │
  │                             │
  └──────────[+]←───────────────┘
            ↓
          ReLU
            ↓
         Output
```

#### 파인튜닝 시 구조 변경

```python
import torch
import torch.nn as nn
from torchvision import models

# 1. 사전 학습 모델 로드
model = models.resnet50(pretrained=True)

# 2. 모델 구조 확인
print(model)
"""
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
  (bn1): BatchNorm2d(64)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1)

  (layer1): Sequential(...)  # 3 blocks
  (layer2): Sequential(...)  # 4 blocks
  (layer3): Sequential(...)  # 6 blocks
  (layer4): Sequential(...)  # 3 blocks

  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=2048, out_features=1000)  ← 이 부분 변경!
)
"""

# 3. 마지막 FC 레이어만 교체 (PPE 2클래스: 안전모 착용/미착용)
num_features = model.fc.in_features  # 2048
model.fc = nn.Linear(num_features, 2)

print(f"Original FC: 2048 → 1000")
print(f"Modified FC: 2048 → 2")

# 4. 변경된 구조
"""
ResNet(
  (conv1): Conv2d(...)
  ...
  (layer4): Sequential(...)
  (avgpool): AdaptiveAvgPool2d(...)
  (fc): Linear(in_features=2048, out_features=2)  ← 변경됨!
)
"""
```

#### 다양한 파인튜닝 전략

**전략 1: 마지막 레이어만 학습 (Feature Extractor)**

```python
# 모든 레이어 동결
for param in model.parameters():
    param.requires_grad = False

# FC 레이어 교체
model.fc = nn.Linear(2048, 2)

# FC 레이어만 학습 가능
for param in model.fc.parameters():
    param.requires_grad = True

# 학습 파라미터 확인
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")  # 약 4,098개
```

**전략 2: 상위 레이어만 학습**

```python
# Layer 1, 2, 3 동결
for name, param in model.named_parameters():
    if 'layer4' not in name and 'fc' not in name:
        param.requires_grad = False

# Layer 4와 FC만 학습
model.fc = nn.Linear(2048, 2)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")  # 약 7.1M개
```

**전략 3: 전체 파인튜닝 (차등 학습률)**

```python
# 모든 레이어 학습 가능
model.fc = nn.Linear(2048, 5)  # 5 클래스

# 레이어별로 다른 학습률 설정
optimizer = torch.optim.Adam([
    {'params': model.conv1.parameters(), 'lr': 1e-6},
    {'params': model.layer1.parameters(), 'lr': 1e-6},
    {'params': model.layer2.parameters(), 'lr': 1e-5},
    {'params': model.layer3.parameters(), 'lr': 1e-5},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
], lr=1e-4)

# 하위 레이어: 낮은 학습률 (이미 잘 학습됨)
# 상위 레이어: 높은 학습률 (작업 특화 필요)
```

**전략 4: 커스텀 헤드 추가**

```python
# 기존 FC 제거 후 복잡한 분류기 추가
model.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 5)  # 5 클래스
)

# 또는 멀티태스크 학습
class MultiTaskHead(nn.Module):
    def __init__(self):
        super().__init__()
        # 안전모 착용 여부 (2클래스)
        self.helmet_head = nn.Linear(2048, 2)
        # 안전조끼 착용 여부 (2클래스)
        self.vest_head = nn.Linear(2048, 2)
        # 작업 유형 (5클래스)
        self.task_head = nn.Linear(2048, 5)

    def forward(self, x):
        helmet = self.helmet_head(x)
        vest = self.vest_head(x)
        task = self.task_head(x)
        return helmet, vest, task

model.fc = MultiTaskHead()
```

---

### 1.2 VGG-16 구조

VGG는 **3×3 작은 필터를 반복 사용**하는 간단하지만 효과적인 구조입니다.

#### 전체 구조

```
Input (224 × 224 × 3)
    ↓
┌─────────────────────────────┐
│ Block 1:                     │
│  - Conv 3×3, 64 filters × 2  │
│  - MaxPool 2×2               │
└─────────────────────────────┘
    ↓ (112 × 112 × 64)
┌─────────────────────────────┐
│ Block 2:                     │
│  - Conv 3×3, 128 filters × 2 │
│  - MaxPool 2×2               │
└─────────────────────────────┘
    ↓ (56 × 56 × 128)
┌─────────────────────────────┐
│ Block 3:                     │
│  - Conv 3×3, 256 filters × 3 │
│  - MaxPool 2×2               │
└─────────────────────────────┘
    ↓ (28 × 28 × 256)
┌─────────────────────────────┐
│ Block 4:                     │
│  - Conv 3×3, 512 filters × 3 │
│  - MaxPool 2×2               │
└─────────────────────────────┘
    ↓ (14 × 14 × 512)
┌─────────────────────────────┐
│ Block 5:                     │
│  - Conv 3×3, 512 filters × 3 │
│  - MaxPool 2×2               │
└─────────────────────────────┘
    ↓ (7 × 7 × 512)
┌─────────────────────────────┐
│ Classifier:                  │
│  - FC 25088 → 4096           │
│  - ReLU + Dropout            │
│  - FC 4096 → 4096            │
│  - ReLU + Dropout            │
│  - FC 4096 → 1000            │
└─────────────────────────────┘
    ↓
Output (1000 classes)
```

#### 파인튜닝 예시

```python
from torchvision import models

model = models.vgg16(pretrained=True)

# VGG는 classifier가 Sequential로 구성됨
print(model.classifier)
"""
Sequential(
  (0): Linear(in_features=25088, out_features=4096)
  (1): ReLU(inplace=True)
  (2): Dropout(p=0.5)
  (3): Linear(in_features=4096, out_features=4096)
  (4): ReLU(inplace=True)
  (5): Dropout(p=0.5)
  (6): Linear(in_features=4096, out_features=1000)  ← 마지막만 변경
)
"""

# 방법 1: 마지막 레이어만 교체
model.classifier[6] = nn.Linear(4096, 5)

# 방법 2: 전체 classifier 교체 (더 가벼운 구조)
model.classifier = nn.Sequential(
    nn.Linear(25088, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 5)
)
```

---

### 1.3 EfficientNet 구조

EfficientNet은 **복합 스케일링(Compound Scaling)**을 사용한 효율적인 모델입니다.

#### 전체 구조 (EfficientNet-B0)

```
Input (224 × 224 × 3)
    ↓
┌─────────────────────────────────┐
│ Stem: Conv 3×3, stride 2, 32ch  │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ MBConv Blocks (Mobile Inverted  │
│ Residual Bottleneck)            │
│                                 │
│ Stage 1: MBConv1, k3×3 × 1      │
│ Stage 2: MBConv6, k3×3 × 2      │
│ Stage 3: MBConv6, k5×5 × 2      │
│ Stage 4: MBConv6, k3×3 × 3      │
│ Stage 5: MBConv6, k5×5 × 3      │
│ Stage 6: MBConv6, k5×5 × 4      │
│ Stage 7: MBConv6, k3×3 × 1      │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ Head: Conv 1×1, 1280ch          │
│ GlobalAvgPool                   │
│ Dropout(0.2)                    │
│ FC 1280 → 1000                  │
└─────────────────────────────────┘
    ↓
Output (1000 classes)
```

#### MBConv Block 구조

```
Input
  ↓
  ├────────────────────────────┐
  │                            │
  │ 1×1 Conv (확장)             │  Expansion
  │ ↓                          │
  │ BatchNorm + Swish          │
  │ ↓                          │
  │ Depthwise Conv             │  Depthwise
  │ ↓                          │
  │ BatchNorm + Swish          │
  │ ↓                          │
  │ SE Module (Squeeze-Excite) │  Attention
  │ ↓                          │
  │ 1×1 Conv (축소)             │  Projection
  │ ↓                          │
  │ BatchNorm                  │
  │                            │
  └───────[+]←─────────────────┘
         ↓
      Output
```

#### 파인튜닝 예시

```python
from torchvision import models

model = models.efficientnet_b0(pretrained=True)

# EfficientNet 구조
print(model)
"""
EfficientNet(
  (features): Sequential(...)    # MBConv blocks
  (avgpool): AdaptiveAvgPool2d(...)
  (classifier): Sequential(
    (0): Dropout(p=0.2)
    (1): Linear(in_features=1280, out_features=1000)
  )
)
"""

# 분류기 교체
model.classifier[1] = nn.Linear(1280, 5)

# 또는 더 복잡한 분류기
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(1280, 256),
    nn.ReLU(),
    nn.Linear(256, 5)
)
```

---

## 2. 객체 탐지 모델

### 2.1 YOLOv8 구조

YOLO는 **One-Stage Detector**로 속도가 빠릅니다.

#### 전체 구조

```
Input Image (640 × 640 × 3)
    ↓
┌──────────────────────────────────────────┐
│ BACKBONE (CSPDarknet)                    │  특징 추출
│                                          │
│  Stem:                                   │
│   - Conv 3×3, 64ch, stride 2             │
│                                          │
│  Stage 1:                                │
│   - Conv 3×3, 128ch, stride 2            │
│   - C2f × 3                              │  ← C2f: CSP Bottleneck
│                                          │
│  Stage 2:                                │
│   - Conv 3×3, 256ch, stride 2            │
│   - C2f × 6                              │
│                                          │
│  Stage 3:                                │
│   - Conv 3×3, 512ch, stride 2            │
│   - C2f × 6                              │
│                                          │
│  Stage 4:                                │
│   - Conv 3×3, 1024ch, stride 2           │
│   - C2f × 3                              │
│   - SPPF (Spatial Pyramid Pooling)       │
└──────────────────────────────────────────┘
    ↓ P3(80×80), P4(40×40), P5(20×20)
┌──────────────────────────────────────────┐
│ NECK (PANet - Path Aggregation Network) │  특징 융합
│                                          │
│  Top-Down Path:                          │
│   - P5 → P4 (Upsample + Concat)          │
│   - P4 → P3 (Upsample + Concat)          │
│                                          │
│  Bottom-Up Path:                         │
│   - P3 → P4 (Downsample + Concat)        │
│   - P4 → P5 (Downsample + Concat)        │
└──────────────────────────────────────────┘
    ↓ Enhanced P3, P4, P5
┌──────────────────────────────────────────┐
│ HEAD (Detection Head)                    │  예측
│                                          │
│  P3 Output (80×80):                      │
│   - Small object detection               │
│   - Conv → (nc + 4 + 16) channels        │
│                                          │
│  P4 Output (40×40):                      │
│   - Medium object detection              │
│   - Conv → (nc + 4 + 16) channels        │
│                                          │
│  P5 Output (20×20):                      │
│   - Large object detection               │
│   - Conv → (nc + 4 + 16) channels        │
│                                          │
│  nc: 클래스 개수                          │
│  4: bbox 좌표 (x, y, w, h)                │
│  16: keypoint (선택사항)                  │
└──────────────────────────────────────────┘
    ↓
Predictions: [boxes, scores, classes]
```

#### C2f Module 구조

```
Input
  ↓
  ├────────────────────┬──────────────┐
  │                    │              │
  │  Conv 1×1 (split)  │              │
  │  ↓                 │              │
  │  ┌──────────┐      │              │
  │  │ Bottleneck│      │              │
  │  └─────┬────┘      │              │
  │        │           │              │
  │  ┌──────────┐      │              │
  │  │ Bottleneck│      │              │
  │  └─────┬────┘      │              │
  │        ↓           │              │
  └────[Concat]←───────┴──────────────┘
          ↓
      Conv 1×1
          ↓
       Output
```

#### 파인튜닝 시 구조 변경

**YOLO는 클래스 개수만 변경하면 됨!**

```python
from ultralytics import YOLO

# 1. 사전 학습 모델 로드 (COCO 80 클래스)
model = YOLO('yolov8s.pt')

# 2. 모델 구조 확인
"""
YOLOv8 구조:
- Backbone: 특징 추출 (변경 없음)
- Neck: 특징 융합 (변경 없음)
- Head: 탐지 헤드 (클래스 개수만 변경)
"""

# 3. 데이터셋 YAML에서 클래스 개수 정의
"""
# ppe_dataset.yaml
nc: 5  # COCO 80 → PPE 5 클래스로 변경
names: ['person', 'helmet', 'no-helmet', 'vest', 'no-vest']
"""

# 4. 학습 시작 - 자동으로 헤드가 재구성됨!
model.train(
    data='ppe_dataset.yaml',  # nc=5 정의됨
    epochs=100
)

# YOLO가 내부적으로 수행하는 작업:
# - Backbone, Neck: 사전 학습 가중치 유지
# - Head: (80 + 4) → (5 + 4)로 자동 변경
#         └─ 클래스   └─ bbox
```

#### YOLO 수동 구조 변경 (고급)

```python
import torch
import torch.nn as nn

# 1. 모델 로드
model = YOLO('yolov8s.pt')

# 2. 헤드 레이어 접근
detect_layer = model.model[-1]  # Detect 레이어

print(f"Original classes: {detect_layer.nc}")  # 80

# 3. 클래스 개수 변경
new_nc = 5
detect_layer.nc = new_nc

# 4. 헤드 Conv 레이어 재구성
for i in range(detect_layer.nl):  # nl: number of detection layers (3)
    old_conv = detect_layer.cv3[i]
    # 기존: (nc + 4) * reg_max → 새로운: (new_nc + 4) * reg_max
    detect_layer.cv3[i] = nn.Conv2d(
        old_conv.in_channels,
        (new_nc + 4) * detect_layer.reg_max,
        kernel_size=1
    )

print(f"Modified classes: {detect_layer.nc}")  # 5
```

#### 레이어 동결 (YOLO)

```python
model = YOLO('yolov8s.pt')

# 방법 1: freeze 파라미터 사용 (간단)
model.train(
    data='ppe_dataset.yaml',
    epochs=100,
    freeze=10  # 처음 10개 레이어 동결
)

# 방법 2: 수동 동결
for idx, (name, param) in enumerate(model.model.named_parameters()):
    if idx < 10:  # Backbone 일부 동결
        param.requires_grad = False
        print(f"Frozen: {name}")

# 방법 3: Backbone 전체 동결, Head만 학습
for name, param in model.model.named_parameters():
    if 'model.22' not in name:  # model.22: Detect 헤드
        param.requires_grad = False
```

---

### 2.2 Faster R-CNN 구조

Faster R-CNN은 **Two-Stage Detector**로 정확도가 높습니다.

#### 전체 구조

```
Input Image
    ↓
┌────────────────────────────────────┐
│ STAGE 1: Region Proposal Network   │
│                                    │
│  Backbone (ResNet-50 FPN):         │
│   - Conv layers                    │
│   - Feature Pyramid Network        │
│   ↓ Feature Maps                   │
│                                    │
│  RPN (Region Proposal Network):    │
│   - Anchor 생성                     │
│   - Objectness score 계산           │
│   - BBox regression                │
│   ↓ ~2000 Region Proposals         │
└────────────────────────────────────┘
    ↓
┌────────────────────────────────────┐
│ STAGE 2: R-CNN                     │
│                                    │
│  ROI Pooling/Align:                │
│   - Proposals를 고정 크기로 변환     │
│   ↓ 7×7×2048 features              │
│                                    │
│  Classification Head:               │
│   - FC 2048 → 1024                 │
│   - FC 1024 → num_classes          │  ← 클래스 분류
│                                    │
│  BBox Regression Head:              │
│   - FC 2048 → 1024                 │
│   - FC 1024 → num_classes × 4      │  ← BBox 좌표
└────────────────────────────────────┘
    ↓
Final Detections
```

#### 파인튜닝 예시

```python
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# 1. 사전 학습 모델 로드 (COCO 91 클래스)
model = fasterrcnn_resnet50_fpn(pretrained=True)

# 2. 분류 헤드 교체
num_classes = 6  # 5 PPE 클래스 + 1 배경

# ROI Head의 box predictor 교체
in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(
    in_features,
    num_classes
)

# 3. 구조 확인
print(model.roi_heads.box_predictor)
"""
FastRCNNPredictor(
  (cls_score): Linear(in_features=1024, out_features=6)      ← 91 → 6
  (bbox_pred): Linear(in_features=1024, out_features=24)     ← 91*4 → 6*4
)
"""

# 4. 학습
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)
```

---

## 3. 세그멘테이션 모델

### 3.1 U-Net 구조

U-Net은 **의료 영상 세그멘테이션**에서 유명한 인코더-디코더 구조입니다.

#### 전체 구조

```
Input (256 × 256 × 3)
    ↓
┌─────────────────────────────────────────────────────────┐
│ ENCODER (Contracting Path)                              │
│                                                         │
│  Level 1:                                               │
│   - Conv 3×3 × 2 → 64ch                                 │
│   - MaxPool 2×2 ─────────────────┐                     │
│   ↓ (128×128×64)                 │ Skip Connection     │
│                                   ↓                     │
│  Level 2:                         ├─────────────────┐   │
│   - Conv 3×3 × 2 → 128ch          │                 │   │
│   - MaxPool 2×2 ──────────────┐   │                 │   │
│   ↓ (64×64×128)               │   │                 │   │
│                               ↓   ↓                 │   │
│  Level 3:                     ├───┼─────────────┐   │   │
│   - Conv 3×3 × 2 → 256ch      │   │             │   │   │
│   - MaxPool 2×2 ───────────┐  │   │             │   │   │
│   ↓ (32×32×256)            │  │   │             │   │   │
│                            ↓  ↓   ↓             │   │   │
│  Level 4:                  ├──┼───┼─────────┐   │   │   │
│   - Conv 3×3 × 2 → 512ch   │  │   │         │   │   │   │
│   - MaxPool 2×2 ────────┐  │  │   │         │   │   │   │
│   ↓ (16×16×512)         │  │  │   │         │   │   │   │
│                         ↓  ↓  ↓   ↓         │   │   │   │
│  Bottleneck:            │  │  │   │         │   │   │   │
│   - Conv 3×3 × 2 → 1024 │  │  │   │         │   │   │   │
│   ↓ (16×16×1024)        │  │  │   │         │   │   │   │
└─────────────────────────┼──┼──┼───┼─────────┼───┼───┼───┘
                          │  │  │   │         │   │   │
┌─────────────────────────┼──┼──┼───┼─────────┼───┼───┼───┐
│ DECODER (Expansive Path)│  │  │   │         │   │   │   │
│                         ↓  │  │   │         │   │   │   │
│  Level 4:               │  │  │   │         │   │   │   │
│   - UpConv 2×2          │  │  │   │         │   │   │   │
│   - Concat ←────────────┘  │  │   │         │   │   │   │
│   - Conv 3×3 × 2 → 512ch   │  │   │         │   │   │   │
│   ↓ (32×32×512)            │  │   │         │   │   │   │
│                            ↓  │   │         │   │   │   │
│  Level 3:                  │  │   │         │   │   │   │
│   - UpConv 2×2             │  │   │         │   │   │   │
│   - Concat ←───────────────┘  │   │         │   │   │   │
│   - Conv 3×3 × 2 → 256ch      │   │         │   │   │   │
│   ↓ (64×64×256)               │   │         │   │   │   │
│                               ↓   │         │   │   │   │
│  Level 2:                     │   │         │   │   │   │
│   - UpConv 2×2                │   │         │   │   │   │
│   - Concat ←──────────────────┘   │         │   │   │   │
│   - Conv 3×3 × 2 → 128ch          │         │   │   │   │
│   ↓ (128×128×128)                 │         │   │   │   │
│                                   ↓         │   │   │   │
│  Level 1:                         │         │   │   │   │
│   - UpConv 2×2                    │         │   │   │   │
│   - Concat ←──────────────────────┘         │   │   │   │
│   - Conv 3×3 × 2 → 64ch                     │   │   │   │
│   ↓ (256×256×64)                            │   │   │   │
│                                             │   │   │   │
│  Output:                                    │   │   │   │
│   - Conv 1×1 → num_classes                  │   │   │   │
└─────────────────────────────────────────────┴───┴───┴───┘
    ↓
Segmentation Map (256 × 256 × num_classes)
```

#### 파인튜닝 예시

```python
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.double_conv(in_channels, 64)
        self.enc2 = self.double_conv(64, 128)
        self.enc3 = self.double_conv(128, 256)
        self.enc4 = self.double_conv(256, 512)

        # Bottleneck
        self.bottleneck = self.double_conv(512, 1024)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.double_conv(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.double_conv(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.double_conv(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.double_conv(128, 64)

        # Output
        self.out = nn.Conv2d(64, num_classes, 1)  ← 클래스 개수만 변경!

    def double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

# 바이너리 세그멘테이션 (안전 구역)
model_binary = UNet(in_channels=3, num_classes=1)

# 멀티클래스 세그멘테이션 (사람, 헬멧, 조끼 등)
model_multi = UNet(in_channels=3, num_classes=5)

# ImageNet 사전 학습 인코더 사용 (더 일반적)
from segmentation_models_pytorch import Unet

model = Unet(
    encoder_name="resnet50",        # ResNet-50 백본
    encoder_weights="imagenet",     # ImageNet 사전 학습
    in_channels=3,
    classes=5,                      # 5 클래스
    activation=None
)

# 인코더 동결 (Feature Extractor)
for param in model.encoder.parameters():
    param.requires_grad = False
```

---

## 4. 파인튜닝 시 구조 변경 패턴

### 4.1 변경하는 부분

```
┌────────────────────────────────────────────────┐
│ 사전 학습 모델                                   │
├────────────────────────────────────────────────┤
│                                                │
│ Input Layer                                    │
│   └─ 입력 채널 수 변경 (예: RGB → 그레이스케일)    │ ← 때때로 변경
│                                                │
│ Feature Extractor (Backbone)                   │
│   └─ Conv, Pooling, Residual Blocks            │ ← 보통 유지
│                                                │
│ Feature Aggregation (Neck)                     │
│   └─ FPN, PANet 등                             │ ← 보통 유지
│                                                │
│ Task-Specific Head                             │
│   └─ 분류/탐지/세그멘테이션 헤드                  │ ← 항상 변경!
│                                                │
│ Output Layer                                   │
│   └─ 클래스 개수에 맞게 조정                      │ ← 항상 변경!
│                                                │
└────────────────────────────────────────────────┘
```

### 4.2 변경 방법 요약

#### 이미지 분류

```python
# ResNet
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(2048, num_classes)  ← 이것만!

# VGG
model = models.vgg16(pretrained=True)
model.classifier[6] = nn.Linear(4096, num_classes)  ← 이것만!

# EfficientNet
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(1280, num_classes)  ← 이것만!
```

#### 객체 탐지

```python
# YOLO
model = YOLO('yolov8s.pt')
# data.yaml에서 nc만 변경 - 자동으로 헤드 재구성!

# Faster R-CNN
model = fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```

#### 세그멘테이션

```python
# U-Net
model = Unet(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    classes=num_classes  ← 클래스 개수만!
)

# DeepLab
model = deeplabv3_resnet50(pretrained=True)
model.classifier[4] = nn.Conv2d(256, num_classes, 1)
```

### 4.3 고급 변경 패턴

#### 멀티태스크 학습

```python
class MultiTaskModel(nn.Module):
    def __init__(self, backbone, num_classes_task1, num_classes_task2):
        super().__init__()

        # 공유 백본 (사전 학습 모델)
        self.backbone = backbone

        # 작업별 헤드
        self.head1 = nn.Linear(2048, num_classes_task1)  # 안전모
        self.head2 = nn.Linear(2048, num_classes_task2)  # 안전조끼

    def forward(self, x):
        features = self.backbone(x)
        out1 = self.head1(features)
        out2 = self.head2(features)
        return out1, out2

# 사용
backbone = models.resnet50(pretrained=True)
backbone.fc = nn.Identity()  # FC 레이어 제거

model = MultiTaskModel(backbone, num_classes_task1=2, num_classes_task2=2)
```

#### 입력 채널 변경 (의료 영상 등)

```python
# 기존: RGB (3채널) → 변경: 단일 채널 (X-ray)
model = models.resnet50(pretrained=True)

# 첫 번째 Conv 레이어 교체
original_conv = model.conv1
model.conv1 = nn.Conv2d(
    1,  # 그레이스케일
    64,
    kernel_size=7,
    stride=2,
    padding=3,
    bias=False
)

# 사전 학습 가중치 재사용 (RGB 평균)
with torch.no_grad():
    model.conv1.weight = nn.Parameter(
        original_conv.weight.mean(dim=1, keepdim=True)
    )

# 또는: 다중 스펙트럼 (위성 사진 등)
# RGB + NIR = 4채널
model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

with torch.no_grad():
    # RGB 채널은 사전 학습 가중치 사용
    model.conv1.weight[:, :3] = original_conv.weight
    # NIR 채널은 랜덤 초기화 또는 평균값 사용
    model.conv1.weight[:, 3:4] = original_conv.weight.mean(dim=1, keepdim=True)
```

#### 중간 특징 활용

```python
class FeatureExtractorModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

        # 중간 레이어 훅 등록
        self.features = {}
        self.backbone.layer2.register_forward_hook(self.get_features('layer2'))
        self.backbone.layer3.register_forward_hook(self.get_features('layer3'))
        self.backbone.layer4.register_forward_hook(self.get_features('layer4'))

    def get_features(self, name):
        def hook(model, input, output):
            self.features[name] = output
        return hook

    def forward(self, x):
        _ = self.backbone(x)
        # 다양한 스케일의 특징 활용
        return self.features['layer2'], self.features['layer3'], self.features['layer4']

# 사용 예: FPN-style 구조
backbone = models.resnet50(pretrained=True)
feature_extractor = FeatureExtractorModel(backbone)
```

---

## 5. 실전 체크리스트

### ✅ 파인튜닝 전 확인사항

```markdown
□ 사전 학습 모델 선택
  - 작업 유형: 분류 / 탐지 / 세그멘테이션
  - 모델 크기: nano < small < medium < large
  - 사전 학습 데이터셋: ImageNet / COCO / OpenImages

□ 입력 형식 확인
  - 입력 채널: RGB(3) / 그레이스케일(1) / 기타
  - 입력 크기: 224×224 / 640×640 / 가변

□ 출력 형식 확인
  - 분류: 클래스 개수
  - 탐지: 클래스 개수 + 박스
  - 세그멘테이션: 픽셀별 클래스

□ 헤드 레이어 교체
  - 마지막 FC / Conv 레이어 확인
  - 새 작업에 맞게 재구성

□ 학습 전략 결정
  - Feature Extractor (헤드만)
  - 부분 파인튜닝 (상위 레이어)
  - 전체 파인튜닝 (모든 레이어)
```

### 🎯 디버깅 팁

```python
# 1. 모델 구조 출력
print(model)

# 2. 레이어별 파라미터 개수
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}, Trainable: {param.requires_grad}")

# 3. 총 파라미터 개수
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total: {total_params:,}")
print(f"Trainable: {trainable_params:,}")
print(f"Frozen: {total_params - trainable_params:,}")

# 4. 입출력 크기 확인
from torchsummary import summary
summary(model, input_size=(3, 224, 224))

# 5. 실제 추론 테스트
dummy_input = torch.randn(1, 3, 224, 224)
output = model(dummy_input)
print(f"Output shape: {output.shape}")
```

---

## 요약

### 핵심 포인트

1. **백본(Backbone)**: 보통 유지 (특징 추출은 범용적)
2. **헤드(Head)**: 항상 변경 (작업 특화)
3. **클래스 개수**: 새 작업에 맞게 조정
4. **학습률**: 헤드는 높게, 백본은 낮게 (차등 학습률)

### 일반적인 파인튜닝 순서

```
1. 사전 학습 모델 로드
   ↓
2. 헤드 레이어 교체 (클래스 개수 변경)
   ↓
3. 레이어 동결 결정 (데이터 양에 따라)
   ↓
4. 학습률 설정 (차등 학습률)
   ↓
5. 학습 시작
   ↓
6. 성능 모니터링 및 조정
```

이제 사전 학습 모델의 구조를 이해하고, 자신의 작업에 맞게 수정할 수 있습니다! 🚀
