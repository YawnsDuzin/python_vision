# Python Vision & Deep Learning for Construction Safety

공사현장 안전관리를 위한 Computer Vision 및 딥러닝 학습 프로젝트입니다.

## 프로젝트 목표

Python을 활용하여 Computer Vision과 딥러닝을 학습하고, 공사현장 안전관리 시스템을 개발하는 것을 목표로 합니다.

## 주요 적용 분야

- **개인보호구(PPE) 탐지**: 안전모, 안전조끼, 안전화 착용 여부 확인
- **위험 구역 침입 감지**: 제한 구역 출입 모니터링
- **작업자 행동 분석**: 위험한 작업 자세 및 행동 감지
- **장비 및 차량 추적**: 중장비 움직임 모니터링
- **낙하물 감지**: 고소 작업 중 낙하물 위험 감지

## 학습 로드맵

상세한 학습 계획은 [LEARNING_ROADMAP.md](./LEARNING_ROADMAP.md)를 참고하세요.

## 레퍼런스

공사현장 안전관리 관련 레퍼런스는 [CONSTRUCTION_SAFETY_REFERENCES.md](./CONSTRUCTION_SAFETY_REFERENCES.md)를 참고하세요.

## 프로젝트 구조

```
python_vision/
├── docs/                      # 문서
├── notebooks/                 # Jupyter 노트북 학습 자료
│   ├── 01_python_basics/     # Python 기초
│   ├── 02_numpy_pandas/      # NumPy & Pandas
│   ├── 03_opencv_basics/     # OpenCV 기초
│   ├── 04_deep_learning/     # 딥러닝 기초
│   └── 05_advanced_vision/   # 고급 Vision
├── projects/                  # 실습 프로젝트
│   ├── beginner/             # 초급 프로젝트
│   ├── intermediate/         # 중급 프로젝트
│   └── advanced/             # 고급 프로젝트
├── datasets/                  # 데이터셋
└── models/                    # 학습된 모델
```

## 시작하기

### 필수 요구사항

- Python 3.8 이상
- CUDA 지원 GPU (권장)

### 설치

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 필수 패키지 설치
pip install -r requirements.txt
```

## 학습 순서

1. **단계 1 (1-2주)**: Python 기초 및 데이터 처리
2. **단계 2 (2-3주)**: OpenCV 및 이미지 처리
3. **단계 3 (3-4주)**: 딥러닝 기초 (PyTorch/TensorFlow)
4. **단계 4 (4-6주)**: Object Detection & Segmentation
5. **단계 5 (6-8주)**: 공사현장 안전관리 프로젝트 구현

## 라이선스

MIT License
