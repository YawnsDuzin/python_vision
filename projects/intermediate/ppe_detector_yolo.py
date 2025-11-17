"""
YOLO 기반 개인보호구(PPE) 탐지 시스템

YOLOv8을 사용하여 안전모와 안전조끼 착용 여부를 실시간으로 탐지합니다.

필수 패키지:
    pip install ultralytics opencv-python

학습 목표:
- YOLO 모델 사용법
- 커스텀 데이터셋 학습
- 실시간 객체 탐지
- 안전 규칙 적용
"""

import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
import yaml
from typing import List, Dict, Tuple


class PPEDetector:
    """
    개인보호구(PPE) 탐지 클래스
    """

    def __init__(self, model_path: str = 'yolov8n.pt', conf_threshold: float = 0.5):
        """
        초기화

        Args:
            model_path: YOLO 모델 경로
            conf_threshold: 신뢰도 임계값
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

        # 클래스 이름 (예시)
        self.class_names = {
            0: 'person',
            1: 'helmet',
            2: 'vest',
            3: 'no-helmet',
            4: 'no-vest'
        }

    def detect(self, image) -> List[Dict]:
        """
        이미지에서 PPE 탐지

        Args:
            image: 입력 이미지 (numpy array)

        Returns:
            검출 결과 리스트
        """
        results = self.model(image, conf=self.conf_threshold)[0]

        detections = []
        for box in results.boxes:
            detection = {
                'class_id': int(box.cls[0]),
                'class_name': self.class_names.get(int(box.cls[0]), 'unknown'),
                'confidence': float(box.conf[0]),
                'bbox': box.xyxy[0].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                'center': self._get_bbox_center(box.xyxy[0].cpu().numpy())
            }
            detections.append(detection)

        return detections

    @staticmethod
    def _get_bbox_center(bbox) -> Tuple[float, float]:
        """바운딩 박스의 중심점 계산"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def check_safety_violations(self, detections: List[Dict]) -> List[Dict]:
        """
        안전 규정 위반 확인

        Args:
            detections: 탐지 결과

        Returns:
            위반 사항 리스트
        """
        violations = []

        # 사람 검출
        persons = [d for d in detections if d['class_name'] == 'person']

        # 안전모 미착용 검사
        no_helmet = [d for d in detections if d['class_name'] == 'no-helmet']
        if no_helmet:
            violations.append({
                'type': 'NO_HELMET',
                'count': len(no_helmet),
                'severity': 'HIGH',
                'message': f'{len(no_helmet)} worker(s) without helmet detected'
            })

        # 안전조끼 미착용 검사
        no_vest = [d for d in detections if d['class_name'] == 'no-vest']
        if no_vest:
            violations.append({
                'type': 'NO_VEST',
                'count': len(no_vest),
                'severity': 'MEDIUM',
                'message': f'{len(no_vest)} worker(s) without vest detected'
            })

        return violations

    def draw_detections(self, image, detections: List[Dict], violations: List[Dict]):
        """
        이미지에 검출 결과 시각화

        Args:
            image: 원본 이미지
            detections: 탐지 결과
            violations: 위반 사항
        """
        result = image.copy()

        # 검출 결과 그리기
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            class_name = det['class_name']
            confidence = det['confidence']

            # 색상 선택 (위반 여부에 따라)
            if 'no-' in class_name:
                color = (0, 0, 255)  # 빨강 (위반)
            else:
                color = (0, 255, 0)  # 초록 (안전)

            # 바운딩 박스
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            # 라벨
            label = f'{class_name}: {confidence:.2f}'
            cv2.putText(
                result,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        # 위반 통계 표시
        y_offset = 30
        for violation in violations:
            text = f"{violation['type']}: {violation['count']}"
            color = (0, 0, 255) if violation['severity'] == 'HIGH' else (0, 165, 255)

            cv2.putText(
                result,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )
            y_offset += 30

        return result


class PPETrainer:
    """
    YOLO 모델 학습 클래스
    """

    def __init__(self, model_name: str = 'yolov8n.pt'):
        """
        초기화

        Args:
            model_name: 사전 학습된 YOLO 모델 이름
        """
        self.model = YOLO(model_name)

    def create_dataset_yaml(self, output_path: str = 'ppe_dataset.yaml'):
        """
        데이터셋 설정 파일 생성

        Args:
            output_path: 출력 YAML 파일 경로
        """
        dataset_config = {
            'path': '../datasets/ppe',  # 데이터셋 루트 경로
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',

            'nc': 5,  # 클래스 개수
            'names': ['person', 'helmet', 'vest', 'no-helmet', 'no-vest']
        }

        with open(output_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)

        print(f"Dataset config saved to {output_path}")

    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        img_size: int = 640,
        batch_size: int = 16,
        device: str = '0'
    ):
        """
        모델 학습

        Args:
            data_yaml: 데이터셋 YAML 파일 경로
            epochs: 학습 에폭 수
            img_size: 이미지 크기
            batch_size: 배치 크기
            device: 디바이스 (cuda:0, cpu 등)
        """
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            workers=8,
            patience=20,
            save=True,
            plots=True,
            name='ppe_detector',

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
            mixup=0.1
        )

        return results

    def validate(self, data_yaml: str):
        """
        모델 검증

        Args:
            data_yaml: 데이터셋 YAML 파일 경로
        """
        metrics = self.model.val(data=data_yaml)

        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall: {metrics.box.mr:.4f}")

        return metrics


def main():
    """
    메인 함수 - 실시간 PPE 탐지 데모
    """
    print("=== PPE Detection System ===")
    print("Loading model...")

    # 탐지기 초기화
    detector = PPEDetector(model_path='yolov8n.pt', conf_threshold=0.5)

    # 웹캠 열기
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    print("Starting detection... Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # PPE 탐지
        detections = detector.detect(frame)

        # 안전 규정 확인
        violations = detector.check_safety_violations(detections)

        # 시각화
        result = detector.draw_detections(frame, detections, violations)

        # 화면 표시
        cv2.imshow('PPE Detection', result)

        # 종료 조건
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def train_custom_model():
    """
    커스텀 데이터로 모델 학습 예시
    """
    print("=== Training Custom PPE Model ===")

    trainer = PPETrainer(model_name='yolov8n.pt')

    # 데이터셋 설정 파일 생성
    trainer.create_dataset_yaml('ppe_dataset.yaml')

    print("\nStarting training...")
    print("Note: Make sure you have prepared the dataset in the correct format")
    print("Dataset structure:")
    print("  datasets/ppe/")
    print("    ├── train/")
    print("    │   ├── images/")
    print("    │   └── labels/")
    print("    ├── val/")
    print("    │   ├── images/")
    print("    │   └── labels/")
    print("    └── test/")
    print("        ├── images/")
    print("        └── labels/")

    # 학습 시작
    # trainer.train(data_yaml='ppe_dataset.yaml', epochs=100)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        train_custom_model()
    else:
        main()
