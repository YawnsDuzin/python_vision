# 고급 프로젝트

통합 건설현장 안전관리 시스템 구현 프로젝트입니다.

## 프로젝트 목록

### 1. 통합 안전 모니터링 시스템
- **디렉토리**: `integrated_safety_system/`
- **기능**:
  - 다중 카메라 지원
  - 실시간 PPE 탐지
  - 객체 추적 (DeepSORT)
  - 위험 구역 침입 감지
  - 자동 알림 시스템
  - 웹 대시보드

### 2. 행동 인식 시스템
- **디렉토리**: `action_recognition/`
- **기능**:
  - Pose Estimation
  - 위험 작업 자세 감지
  - 낙상 감지
  - 이상 행동 탐지

### 3. 데이터 분석 플랫폼
- **디렉토리**: `analytics_platform/`
- **기능**:
  - 일일/주간/월간 리포트 생성
  - 통계 분석
  - 위험 패턴 분석
  - 예측 모델

## 시스템 아키텍처

```
┌─────────────┐
│  IP Cameras │
└──────┬──────┘
       │
       v
┌─────────────────┐
│ Stream Handler  │
│ (RTSP/ONVIF)   │
└──────┬──────────┘
       │
       v
┌─────────────────┐
│ Preprocessing   │
│ (Resize, Norm)  │
└──────┬──────────┘
       │
       v
┌─────────────────┐
│ Detection       │
│ (YOLO/Faster)   │
└──────┬──────────┘
       │
       v
┌─────────────────┐
│ Tracking        │
│ (DeepSORT)      │
└──────┬──────────┘
       │
       v
┌─────────────────┐
│ Rules Engine    │
│ (Safety Checks) │
└──────┬──────────┘
       │
       ├─────────────┬─────────────┬──────────────┐
       v             v             v              v
┌───────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
│  Alert    │ │ Database │ │ Dashboard│ │ API Server   │
│  System   │ │          │ │ (Web UI) │ │ (REST/WS)    │
└───────────┘ └──────────┘ └──────────┘ └──────────────┘
```

## 기술 스택

### Backend
- **Framework**: FastAPI
- **AI/ML**: PyTorch, Ultralytics
- **Database**: PostgreSQL, Redis
- **Message Queue**: RabbitMQ or Kafka
- **Monitoring**: Prometheus, Grafana

### Frontend
- **Dashboard**: Streamlit or React
- **Visualization**: Plotly, D3.js

### Deployment
- **Container**: Docker, Docker Compose
- **Orchestration**: Kubernetes (optional)
- **Cloud**: AWS, GCP, Azure
- **Edge**: NVIDIA Jetson

## 시작하기

### 필수 요구사항
- Python 3.8+
- CUDA 11.8+
- Docker & Docker Compose
- NVIDIA GPU (권장: RTX 3060 이상)

### 설치

```bash
# 1. 가상환경 생성
python -m venv venv
source venv/bin/activate

# 2. 패키지 설치
pip install -r requirements.txt

# 3. 환경 변수 설정
cp .env.example .env
# .env 파일을 편집하여 설정 조정

# 4. 데이터베이스 초기화
python scripts/init_db.py

# 5. 모델 다운로드
python scripts/download_models.py
```

### 실행

```bash
# 개발 모드
python main.py --mode dev

# 프로덕션 모드 (Docker)
docker-compose up -d
```

## 프로젝트 구조

```
integrated_safety_system/
├── src/
│   ├── capture/          # 카메라 스트림 처리
│   ├── detection/        # 객체 탐지
│   ├── tracking/         # 객체 추적
│   ├── analysis/         # 행동 분석
│   ├── rules/            # 안전 규칙 엔진
│   ├── alert/            # 알림 시스템
│   ├── api/              # REST API
│   └── dashboard/        # 웹 대시보드
├── models/               # 학습된 모델
├── configs/              # 설정 파일
├── tests/                # 테스트 코드
├── docker/               # Docker 파일
└── docs/                 # 문서
```

## 개발 가이드

### 새 카메라 추가
1. `configs/cameras.yaml`에 카메라 정보 추가
2. 시스템 재시작

### 새 규칙 추가
1. `src/rules/custom_rules.py`에 규칙 정의
2. `configs/rules.yaml`에 규칙 활성화

### API 엔드포인트
- `GET /api/cameras` - 카메라 목록
- `GET /api/detections` - 최근 탐지 결과
- `GET /api/violations` - 위반 사항
- `POST /api/cameras/{id}/snapshot` - 스냅샷 촬영

## 라이선스
MIT License
