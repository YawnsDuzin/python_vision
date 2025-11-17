# 시작 가이드

Python Vision 학습을 시작하는 방법을 단계별로 안내합니다.

## 1. 환경 설정

### Python 설치 확인
```bash
python --version
# Python 3.8 이상 필요
```

### 가상환경 생성
```bash
# 가상환경 생성
python -m venv venv

# 활성화
# Linux/Mac:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### 패키지 설치
```bash
# 기본 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt

# GPU 지원 (CUDA 사용 가능한 경우)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 2. 학습 경로

### Week 1: Python 기초
**목표**: Python 프로그래밍 기본 마스터

1. **기본 문법 학습** (2일)
   - 변수, 데이터 타입, 연산자
   - 조건문, 반복문
   - 함수

2. **데이터 구조** (2일)
   - 리스트, 딕셔너리, 튜플, 집합
   - 리스트 컴프리헨션
   - 파일 입출력

3. **NumPy & Pandas** (3일)
   - NumPy 배열 조작
   - Pandas DataFrame
   - 데이터 분석 실습

**실습 프로젝트**:
```bash
# 안전 데이터 분석
python notebooks/01_python_basics/safety_data_analysis.py
```

### Week 2: OpenCV & 이미지 처리
**목표**: 이미지 처리 기본 기술 습득

1. **OpenCV 기초** (3일)
   - 이미지 읽기/쓰기
   - 색상 공간 변환
   - 이미지 연산

2. **이미지 전처리** (2일)
   - 필터링, 블러링
   - 엣지 검출
   - 이진화

3. **객체 검출 기초** (2일)
   - 컨투어 검출
   - 색상 기반 검출

**실습 프로젝트**:
```bash
# 안전조끼 검출기
python projects/beginner/01_color_detection.py
```

### Week 3: 딥러닝 기초
**목표**: 딥러닝 개념 이해 및 PyTorch 학습

1. **딥러닝 이론** (2일)
   - 신경망 기초
   - 역전파 알고리즘
   - 손실 함수와 최적화

2. **PyTorch 실습** (3일)
   - 텐서 연산
   - 간단한 신경망 구현
   - 모델 학습 및 평가

3. **CNN** (2일)
   - CNN 구조 이해
   - 이미지 분류 실습

**실습 프로젝트**:
```bash
# 안전모 분류기
python notebooks/04_deep_learning/helmet_classifier.py
```

### Week 4-5: Object Detection
**목표**: YOLO 기반 객체 탐지 시스템 구현

1. **YOLO 이론** (2일)
   - YOLO 아키텍처
   - 평가 지표 (IoU, mAP)

2. **YOLOv8 실습** (3일)
   - 모델 학습
   - 커스텀 데이터셋 준비
   - 추론 및 평가

3. **실시간 탐지** (3일)
   - 비디오 처리
   - 웹캠 연동
   - 성능 최적화

**실습 프로젝트**:
```bash
# PPE 탐지 시스템
python projects/intermediate/ppe_detector_yolo.py
```

### Week 6-8: 통합 프로젝트
**목표**: 실무 수준의 안전관리 시스템 구축

1. **시스템 설계** (3일)
   - 아키텍처 설계
   - 데이터베이스 설계
   - API 설계

2. **핵심 기능 구현** (10일)
   - 다중 카메라 지원
   - 실시간 탐지 및 추적
   - 규칙 엔진
   - 알림 시스템

3. **대시보드 개발** (5일)
   - 웹 인터페이스
   - 통계 시각화
   - 리포트 생성

4. **배포** (3일)
   - Docker 컨테이너화
   - 클라우드/엣지 배포
   - 모니터링 설정

## 3. 일일 학습 루틴

### 평일 (2-3시간)
1. **이론 학습** (30분)
   - 강의 영상 시청
   - 문서 읽기

2. **코딩 실습** (1-1.5시간)
   - 예제 코드 실행
   - 직접 코드 작성

3. **복습 및 정리** (30분)
   - 학습 내용 정리
   - 문제 해결 연습

### 주말 (4-6시간)
1. **프로젝트 작업** (3-4시간)
   - 주간 프로젝트 완성
   - 코드 리팩토링

2. **심화 학습** (1-2시간)
   - 논문 읽기
   - 추가 자료 학습

## 4. 학습 자료

### 온라인 강의
- [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)
- [Fast.ai Practical Deep Learning](https://www.fast.ai/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

### 유튜브 채널
- [Nicolai Nielsen](https://www.youtube.com/@NicolaiAI)
- [Augmented Startups](https://www.youtube.com/@AugmentedStartups)

### 블로그
- [PyImageSearch](https://www.pyimagesearch.com/)
- [Towards Data Science](https://towardsdatascience.com/)

### 책
- "Deep Learning with Python" by François Chollet
- "Hands-On Machine Learning" by Aurélien Géron
- "Python for Computer Vision" by Adrian Rosebrock

## 5. 실습 데이터셋

### 다운로드 링크
```bash
# Hard Hat 데이터셋
wget https://public.roboflow.com/ds/dataset_id -O hardhat_dataset.zip
unzip hardhat_dataset.zip -d datasets/hardhat

# PPE 데이터셋
# Kaggle에서 다운로드: https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection
```

### 데이터셋 구조
```
datasets/
├── hardhat/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
```

## 6. 개발 도구

### IDE 추천
- **VSCode**: 가볍고 확장성 좋음
- **PyCharm**: Python 전용 IDE
- **Jupyter Lab**: 데이터 분석 및 실험

### VSCode 확장 프로그램
- Python
- Pylance
- Jupyter
- GitLens
- Docker

### 디버깅 팁
```python
# 이미지 확인
import cv2
cv2.imshow('Debug', image)
cv2.waitKey(0)

# 변수 확인
print(f"Shape: {image.shape}, Type: {image.dtype}")

# 중단점
import pdb; pdb.set_trace()
```

## 7. 문제 해결

### GPU 사용 확인
```python
import torch

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Count: {torch.cuda.device_count()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
```

### 메모리 부족 문제
```python
# 배치 크기 줄이기
batch_size = 8  # 16에서 8로 감소

# 이미지 크기 줄이기
imgsz = 416  # 640에서 416으로 감소

# 혼합 정밀도 사용
model.train(amp=True)
```

### 속도 개선
```python
# 모델 양자화
model.export(format='onnx', half=True)

# 배치 추론
results = model(images, batch=16)
```

## 8. 커뮤니티 참여

### GitHub
- 자신의 프로젝트 공개
- 이슈 제기 및 PR 제출
- 다른 프로젝트 참고

### Kaggle
- 대회 참여
- 노트북 공유
- 토론 참여

### Discord/Slack
- Computer Vision 커뮤니티 가입
- 질문 및 토론

## 9. 학습 체크리스트

### 단계 1 체크리스트
- [ ] Python 기본 문법 완료
- [ ] NumPy 배열 다루기
- [ ] Pandas DataFrame 사용
- [ ] 데이터 시각화
- [ ] 실습 프로젝트 완성

### 단계 2 체크리스트
- [ ] OpenCV 이미지 읽기/쓰기
- [ ] 색상 공간 변환
- [ ] 필터링 및 전처리
- [ ] 컨투어 검출
- [ ] 비디오 처리

### 단계 3 체크리스트
- [ ] PyTorch 텐서 연산
- [ ] 신경망 구현
- [ ] 모델 학습 및 평가
- [ ] CNN 구조 이해
- [ ] 전이 학습 적용

### 단계 4 체크리스트
- [ ] YOLO 모델 사용
- [ ] 커스텀 데이터셋 준비
- [ ] 모델 학습
- [ ] 실시간 탐지 구현
- [ ] 성능 최적화

### 단계 5 체크리스트
- [ ] 시스템 아키텍처 설계
- [ ] 핵심 기능 구현
- [ ] API 개발
- [ ] 웹 대시보드
- [ ] 배포 완료

## 10. 다음 단계

학습 완료 후:
1. **포트폴리오 구축**
   - GitHub에 프로젝트 정리
   - README 작성
   - 데모 영상 제작

2. **심화 학습**
   - 최신 논문 읽기
   - 대회 참여
   - 오픈소스 기여

3. **실무 적용**
   - 실제 프로젝트 수주
   - 스타트업 도전
   - 연구 참여

**건승을 기원합니다! 🚀**
