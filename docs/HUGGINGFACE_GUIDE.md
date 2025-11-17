# Hugging Face 모델 사용 가이드 (Computer Vision)

Hugging Face Hub의 사전 학습 모델을 활용하는 방법과 일반적인 Vision 작업을 정리합니다.

---

## 목차
1. [Hugging Face란?](#1-hugging-face란)
2. [설치 및 기본 사용법](#2-설치-및-기본-사용법)
3. [일반적인 Vision 작업](#3-일반적인-vision-작업)
4. [파인튜닝 방법](#4-파인튜닝-방법)
5. [모델 공유 및 배포](#5-모델-공유-및-배포)

---

## 1. Hugging Face란?

### 개요
**Hugging Face Hub**는 사전 학습된 AI 모델을 공유하고 사용할 수 있는 플랫폼입니다.

```
┌─────────────────────────────────────────┐
│        Hugging Face Hub                 │
├─────────────────────────────────────────┤
│                                         │
│ 🤗 Models: 400,000+ 사전 학습 모델       │
│   - NLP: BERT, GPT, T5, LLaMA 등        │
│   - Vision: ViT, DETR, YOLO 등          │
│   - Multimodal: CLIP, BLIP 등           │
│                                         │
│ 📦 Datasets: 100,000+ 데이터셋           │
│   - ImageNet, COCO, CelebA 등           │
│                                         │
│ 🚀 Spaces: 데모 및 앱 호스팅             │
│   - Gradio, Streamlit 앱                │
│                                         │
└─────────────────────────────────────────┘
```

### 장점
1. **즉시 사용 가능**: 사전 학습 모델을 몇 줄의 코드로 사용
2. **통합 API**: 모든 모델이 동일한 인터페이스
3. **버전 관리**: Git 기반 모델 버전 관리
4. **무료 호스팅**: 모델 및 데모 무료 호스팅
5. **커뮤니티**: 활발한 커뮤니티와 문서

### vs 기존 방법

| 항목 | 기존 방법 (torchvision 등) | Hugging Face |
|------|---------------------------|--------------|
| **모델 수** | 수십 개 | 400,000+ |
| **다운로드** | 수동 스크립트 | 자동 캐싱 |
| **버전 관리** | 직접 관리 | Git 기반 자동 |
| **파인튜닝** | 직접 코드 작성 | Trainer API |
| **공유** | 별도 서버 필요 | Hub에 자동 업로드 |

---

## 2. 설치 및 기본 사용법

### 2.1 설치

```bash
# Transformers (핵심 라이브러리)
pip install transformers

# 추가 패키지
pip install transformers[torch]  # PyTorch
pip install transformers[vision]  # Vision 관련
pip install datasets  # 데이터셋 로드
pip install accelerate  # 학습 가속화
pip install timm  # Vision 모델 추가
```

### 2.2 기본 사용법 (3줄 코드)

#### 이미지 분류

```python
from transformers import pipeline

# 1. 파이프라인 생성 (자동으로 모델 다운로드)
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

# 2. 추론
result = classifier("construction_site.jpg")

# 3. 결과
print(result)
"""
[
    {'label': 'construction site', 'score': 0.95},
    {'label': 'building', 'score': 0.03},
    {'label': 'outdoor', 'score': 0.02}
]
"""
```

#### 객체 탐지

```python
from transformers import pipeline

# DETR (DEtection TRansformer) 사용
detector = pipeline("object-detection", model="facebook/detr-resnet-50")

result = detector("image.jpg")

print(result)
"""
[
    {'label': 'person', 'score': 0.99, 'box': {'xmin': 100, 'ymin': 200, ...}},
    {'label': 'hardhat', 'score': 0.95, 'box': {'xmin': 120, 'ymin': 180, ...}}
]
"""
```

### 2.3 모델 검색 및 선택

#### 웹에서 검색
```
https://huggingface.co/models

필터:
- Task: Image Classification / Object Detection / Image Segmentation
- Library: PyTorch / TensorFlow
- License: Apache 2.0 / MIT
- Sort: Most downloads / Trending
```

#### 코드로 검색

```python
from huggingface_hub import list_models

# Vision 모델 검색
models = list_models(
    filter="image-classification",
    sort="downloads",
    limit=10
)

for model in models:
    print(f"{model.modelId} - Downloads: {model.downloads}")

"""
google/vit-base-patch16-224 - Downloads: 15M
microsoft/resnet-50 - Downloads: 8M
facebook/deit-base-distilled-patch16-224 - Downloads: 5M
...
"""
```

---

## 3. 일반적인 Vision 작업

### 3.1 이미지 분류 (Image Classification)

**사용 사례**: 안전모 착용 여부 판별, 작업 유형 분류

#### Vision Transformer (ViT)

```python
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch

# 1. 모델 및 프로세서 로드
model_name = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# 2. 이미지 로드
image = Image.open("worker.jpg")

# 3. 전처리
inputs = processor(images=image, return_tensors="pt")

# 4. 추론
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# 5. 결과
predicted_class_idx = logits.argmax(-1).item()
print(f"Predicted class: {model.config.id2label[predicted_class_idx]}")
```

#### 파인튜닝 (커스텀 데이터)

```python
from transformers import ViTImageProcessor, ViTForImageClassification, Trainer, TrainingArguments
from datasets import load_dataset

# 1. 데이터셋 로드
dataset = load_dataset("imagefolder", data_dir="./ppe_dataset")

# 데이터셋 구조:
# ppe_dataset/
#   ├── train/
#   │   ├── helmet/
#   │   │   ├── img1.jpg
#   │   │   └── img2.jpg
#   │   └── no-helmet/
#   │       ├── img3.jpg
#   │       └── img4.jpg
#   └── test/
#       └── ...

# 2. 전처리 함수
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

def transform(examples):
    inputs = processor(examples['image'], return_tensors='pt')
    inputs['labels'] = examples['label']
    return inputs

dataset = dataset.with_transform(transform)

# 3. 모델 로드 (클래스 개수 지정)
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=2,  # helmet, no-helmet
    ignore_mismatched_sizes=True  # 헤드 크기 불일치 무시
)

# 4. 학습 설정
training_args = TrainingArguments(
    output_dir="./vit-ppe-classifier",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    learning_rate=2e-5,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,  # Hub에 자동 업로드 (선택)
)

# 5. Trainer로 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

trainer.train()

# 6. 모델 저장
trainer.save_model("./vit-ppe-final")
```

---

### 3.2 객체 탐지 (Object Detection)

**사용 사례**: PPE 탐지, 작업자 및 장비 위치 파악

#### DETR (DEtection TRansformer)

```python
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw

# 1. 모델 로드
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# 2. 이미지 로드
image = Image.open("construction_site.jpg")

# 3. 전처리
inputs = processor(images=image, return_tensors="pt")

# 4. 추론
with torch.no_grad():
    outputs = model(**inputs)

# 5. 후처리
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(
    outputs,
    target_sizes=target_sizes,
    threshold=0.5
)[0]

# 6. 결과 시각화
draw = ImageDraw.Draw(image)

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    label_text = model.config.id2label[label.item()]

    # 박스 그리기
    draw.rectangle(box, outline="red", width=3)
    draw.text((box[0], box[1]), f"{label_text}: {round(score.item(), 2)}", fill="red")

image.save("result.jpg")
```

#### DETR 파인튜닝

```python
from transformers import DetrImageProcessor, DetrForObjectDetection
from datasets import load_dataset
import torch

# 1. 커스텀 데이터셋 (COCO 형식)
"""
annotations.json:
{
    "images": [{"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480}],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [100, 200, 50, 60],  # [x, y, width, height]
            "area": 3000
        }
    ],
    "categories": [
        {"id": 1, "name": "helmet"},
        {"id": 2, "name": "vest"}
    ]
}
"""

# 2. 데이터셋 로드
dataset = load_dataset("json", data_files={"train": "annotations.json"})

# 3. 모델 로드
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=2,  # helmet, vest
    ignore_mismatched_sizes=True
)

# 4. 전처리 함수
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

def transform(batch):
    # DETR 형식으로 변환
    images = [Image.open(img['file_name']) for img in batch['images']]
    annotations = batch['annotations']

    encoding = processor(
        images=images,
        annotations=annotations,
        return_tensors="pt"
    )
    return encoding

# 5. Trainer로 학습 (위와 동일)
```

---

### 3.3 이미지 세그멘테이션 (Image Segmentation)

**사용 사례**: 위험 구역 식별, 작업 영역 분할

#### SegFormer

```python
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import torch
import numpy as np

# 1. 모델 로드
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

# 2. 이미지 로드
image = Image.open("site.jpg")

# 3. 전처리
inputs = processor(images=image, return_tensors="pt")

# 4. 추론
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# 5. 후처리
upsampled_logits = torch.nn.functional.interpolate(
    logits,
    size=image.size[::-1],
    mode="bilinear",
    align_corners=False,
)

pred_seg = upsampled_logits.argmax(dim=1)[0]

# 6. 시각화
import matplotlib.pyplot as plt

plt.imshow(pred_seg.cpu().numpy())
plt.colorbar()
plt.savefig("segmentation.jpg")
```

---

### 3.4 Zero-Shot 분류 (CLIP)

**사용 사례**: 라벨 없이 텍스트로 이미지 분류

#### CLIP 사용법

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# 1. 모델 로드
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 2. 이미지 및 텍스트
image = Image.open("worker.jpg")
text_labels = [
    "a worker wearing a helmet",
    "a worker without a helmet",
    "a worker wearing a safety vest",
    "a construction site"
]

# 3. 전처리
inputs = processor(
    text=text_labels,
    images=image,
    return_tensors="pt",
    padding=True
)

# 4. 추론
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # (1, 4)
probs = logits_per_image.softmax(dim=1)  # 확률

# 5. 결과
for label, prob in zip(text_labels, probs[0]):
    print(f"{label}: {prob.item():.2%}")

"""
a worker wearing a helmet: 65.3%
a worker without a helmet: 15.2%
a worker wearing a safety vest: 12.1%
a construction site: 7.4%
"""
```

**Zero-Shot 객체 탐지 (OWL-ViT)**

```python
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image, ImageDraw

# 1. 모델 로드
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

# 2. 이미지 및 텍스트 쿼리
image = Image.open("site.jpg")
texts = [["a helmet", "a safety vest", "a person"]]

# 3. 전처리
inputs = processor(text=texts, images=image, return_tensors="pt")

# 4. 추론
outputs = model(**inputs)

# 5. 후처리
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(
    outputs=outputs,
    target_sizes=target_sizes,
    threshold=0.1
)[0]

# 6. 시각화
draw = ImageDraw.Draw(image)

for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
    box = [round(i, 2) for i in box.tolist()]
    draw.rectangle(box, outline="red", width=2)
    draw.text((box[0], box[1]), f"{texts[0][label]}: {score:.2f}", fill="red")

image.save("zero_shot_detection.jpg")
```

---

### 3.5 비전-언어 모델 (Vision-Language)

**사용 사례**: 이미지 캡셔닝, VQA (Visual Question Answering)

#### 이미지 캡셔닝 (BLIP)

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# 1. 모델 로드
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 2. 이미지 로드
image = Image.open("construction.jpg")

# 3. 캡션 생성
inputs = processor(image, return_tensors="pt")
outputs = model.generate(**inputs)
caption = processor.decode(outputs[0], skip_special_tokens=True)

print(caption)
# "a construction worker wearing a yellow hard hat and safety vest at a building site"
```

#### Visual Question Answering (VQA)

```python
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image

# 1. 모델 로드
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# 2. 이미지와 질문
image = Image.open("worker.jpg")
question = "Is the worker wearing a helmet?"

# 3. 추론
inputs = processor(image, question, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
idx = logits.argmax(-1).item()

answer = model.config.id2label[idx]
print(f"Q: {question}")
print(f"A: {answer}")
# A: yes
```

---

## 4. 파인튜닝 방법

### 4.1 Trainer API 사용 (권장)

**장점**: 간단하고 표준화된 학습 파이프라인

```python
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score

# 1. 데이터셋 로드
dataset = load_dataset("imagefolder", data_dir="./data")

# 2. 전처리
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

def preprocess(examples):
    inputs = processor(examples['image'], return_tensors='pt')
    inputs['labels'] = examples['label']
    return inputs

dataset = dataset.with_transform(preprocess)

# 3. 모델 로드
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=len(dataset['train'].features['label'].names),
    ignore_mismatched_sizes=True
)

# 4. 평가 메트릭
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        'accuracy': accuracy_score(labels, predictions)
    }

# 5. 학습 설정
training_args = TrainingArguments(
    output_dir="./results",

    # 학습 파라미터
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,

    # 로깅
    logging_dir="./logs",
    logging_steps=10,

    # 평가
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",

    # 기타
    remove_unused_columns=False,
    push_to_hub=False,
)

# 6. Trainer 생성 및 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)

# 학습 시작
trainer.train()

# 평가
metrics = trainer.evaluate()
print(metrics)

# 저장
trainer.save_model("./final_model")
```

### 4.2 커스텀 학습 루프

**유연성이 필요한 경우**

```python
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from torch.utils.data import DataLoader
from tqdm import tqdm

# 모델 및 데이터 로드
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=2
)
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

# DataLoader
train_loader = DataLoader(dataset['train'], batch_size=16, shuffle=True)

# 옵티마이저
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# 학습 루프
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(10):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader):
        # 데이터를 GPU로
        inputs = {k: v.to(device) for k, v in batch.items()}

        # Forward
        outputs = model(**inputs)
        loss = outputs.loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# 저장
model.save_pretrained("./custom_model")
```

---

## 5. 모델 공유 및 배포

### 5.1 Hugging Face Hub에 업로드

```python
from transformers import ViTForImageClassification
from huggingface_hub import notebook_login

# 1. 로그인 (처음 한 번만)
notebook_login()  # 또는 huggingface-cli login

# 2. 모델 로드
model = ViTForImageClassification.from_pretrained("./final_model")

# 3. Hub에 푸시
model.push_to_hub("username/ppe-helmet-detector")

# 4. 프로세서도 업로드
processor.push_to_hub("username/ppe-helmet-detector")
```

### 5.2 다른 사람이 사용하는 방법

```python
from transformers import pipeline

# 단 한 줄로 사용 가능!
detector = pipeline("image-classification", model="username/ppe-helmet-detector")

result = detector("worker.jpg")
print(result)
```

### 5.3 모델 카드 작성

```markdown
---
# README.md (자동 생성)

language: en
license: apache-2.0
tags:
- vision
- image-classification
- construction-safety
datasets:
- custom
metrics:
- accuracy
---

# PPE Helmet Detector

## Model Description
This model detects whether construction workers are wearing helmets.

## Training Data
- 5,000 images of construction workers
- 2 classes: helmet, no-helmet

## Performance
- Accuracy: 95.3%
- Precision: 94.8%
- Recall: 96.1%

## Usage
```python
from transformers import pipeline

detector = pipeline("image-classification", model="username/ppe-helmet-detector")
result = detector("worker.jpg")
```

## Limitations
- Works best with clear frontal views
- May struggle with partially visible helmets
```

---

## 6. 일반적인 작업 요약

### Vision 작업별 추천 모델

| 작업 | 추천 모델 | 사용 사례 |
|------|----------|----------|
| **이미지 분류** | ViT, DeiT, Swin | 안전모 착용 여부 |
| **객체 탐지** | DETR, YOLOS | PPE 탐지 |
| **세그멘테이션** | SegFormer, Mask2Former | 위험 구역 식별 |
| **Zero-Shot** | CLIP, OWL-ViT | 라벨 없는 분류/탐지 |
| **VQA** | ViLT, BLIP | "안전모 착용했나요?" |

### Hugging Face vs 기존 라이브러리

| 상황 | 추천 |
|------|------|
| **빠른 프로토타입** | ✅ Hugging Face (pipeline) |
| **최신 Transformer 모델** | ✅ Hugging Face |
| **실시간 추론 (속도 중요)** | ⚠️ YOLO, TensorRT |
| **Edge 디바이스** | ⚠️ YOLO, MobileNet |
| **연구/실험** | ✅ Hugging Face |
| **프로덕션 (안정성)** | 둘 다 OK |

---

## 7. 실전 예시: PPE 탐지 시스템

### 전체 워크플로우

```python
from transformers import (
    DetrImageProcessor,
    DetrForObjectDetection,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
import torch

# ===== 1. 데이터셋 준비 =====
# COCO 형식 annotations.json 준비

# ===== 2. 데이터셋 로드 =====
dataset = load_dataset("coco", data_dir="./ppe_coco")

# ===== 3. 모델 로드 및 수정 =====
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=5,  # person, helmet, no-helmet, vest, no-vest
    ignore_mismatched_sizes=True
)

# ===== 4. 학습 =====
training_args = TrainingArguments(
    output_dir="./ppe-detr",
    num_train_epochs=50,
    per_device_train_batch_size=4,
    learning_rate=1e-5,
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
)

trainer.train()

# ===== 5. Hub에 업로드 =====
model.push_to_hub("your-username/ppe-detector")

# ===== 6. 사용 =====
from transformers import pipeline

detector = pipeline("object-detection", model="your-username/ppe-detector")
results = detector("construction_site.jpg")

for detection in results:
    print(f"{detection['label']}: {detection['score']:.2f}")
```

---

## 8. 고급 기능

### 8.1 Mixed Precision Training (속도 향상)

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    fp16=True,  # ← Mixed precision
    per_device_train_batch_size=32,  # 배치 크기 증가 가능
)
```

### 8.2 Gradient Accumulation (메모리 부족 시)

```python
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # 실제 배치 크기 = 4 × 4 = 16
)
```

### 8.3 분산 학습 (Multi-GPU)

```bash
# 명령줄에서
accelerate launch train.py
```

```python
# train.py
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    # 자동으로 모든 GPU 사용
)
```

### 8.4 모델 양자화 (추론 속도 향상)

```python
from transformers import AutoModelForImageClassification
import torch

# 모델 로드
model = AutoModelForImageClassification.from_pretrained("model_name")

# Dynamic Quantization
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# 추론 속도 2-4배 향상, 모델 크기 75% 감소
```

---

## 9. 문제 해결

### 자주 발생하는 오류

#### 1. 모델 다운로드 실패
```python
# 해결: 오프라인 모드 또는 로컬 경로
model = AutoModel.from_pretrained("./local_model")
```

#### 2. GPU 메모리 부족
```python
# 해결: 배치 크기 줄이기 + gradient accumulation
training_args = TrainingArguments(
    per_device_train_batch_size=2,  # 줄이기
    gradient_accumulation_steps=8,  # 늘리기
)
```

#### 3. 클래스 개수 불일치
```python
# 해결: ignore_mismatched_sizes=True
model = ViTForImageClassification.from_pretrained(
    "model_name",
    num_labels=5,
    ignore_mismatched_sizes=True  # ← 추가
)
```

---

## 10. 참고 자료

### 공식 문서
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets)
- [Hugging Face Hub](https://huggingface.co/docs/hub)

### 튜토리얼
- [Vision Transformers Tutorial](https://huggingface.co/docs/transformers/tasks/image_classification)
- [Object Detection Tutorial](https://huggingface.co/docs/transformers/tasks/object_detection)
- [Fine-tuning Guide](https://huggingface.co/docs/transformers/training)

### 커뮤니티
- [Hugging Face Forums](https://discuss.huggingface.co/)
- [Discord](https://hf.co/join/discord)

---

## 요약

### Hugging Face를 사용하는 이유

1. **즉시 사용**: `pipeline()` 한 줄로 사전 학습 모델 사용
2. **통합 API**: 모든 모델이 동일한 인터페이스
3. **최신 모델**: Transformer 기반 SOTA 모델
4. **쉬운 파인튜닝**: Trainer API로 간단하게 학습
5. **공유 및 배포**: Hub에 업로드하면 누구나 사용 가능

### 일반적인 사용 패턴

```python
# 1. 빠른 추론 (프로토타입)
from transformers import pipeline
pipe = pipeline("task", model="model_name")
result = pipe("input")

# 2. 파인튜닝 (프로덕션)
from transformers import Trainer, TrainingArguments
trainer = Trainer(model, args, train_dataset, eval_dataset)
trainer.train()

# 3. 공유
model.push_to_hub("username/model_name")
```

**Hugging Face는 Vision Transformer 시대의 필수 도구입니다!** 🤗
