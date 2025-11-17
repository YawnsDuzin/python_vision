"""
색상 기반 안전조끼 검출기

OpenCV를 사용하여 HSV 색상 공간에서 주황색 안전조끼를 검출합니다.

학습 목표:
- OpenCV 기본 사용법
- 색상 공간 변환
- 이미지 마스킹
- 컨투어 검출
"""

import cv2
import numpy as np


def detect_safety_vest(image_path):
    """
    이미지에서 주황색 안전조끼를 검출합니다.

    Args:
        image_path (str): 입력 이미지 경로

    Returns:
        tuple: (원본 이미지, 검출 결과 이미지, 검출 개수)
    """
    # 이미지 읽기
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image from {image_path}")
        return None, None, 0

    # HSV 색상 공간으로 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 주황색 범위 정의 (안전조끼)
    # HSV에서 주황색 범위
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([15, 255, 255])

    # 마스크 생성
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # 노이즈 제거 (모폴로지 연산)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 컨투어 검출
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 결과 이미지 생성
    result = image.copy()

    # 검출된 조끼 개수
    vest_count = 0

    # 각 컨투어 처리
    for contour in contours:
        # 작은 영역 무시
        area = cv2.contourArea(contour)
        if area < 500:
            continue

        # 바운딩 박스 그리기
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 라벨 추가
        cv2.putText(
            result,
            f'Safety Vest #{vest_count + 1}',
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        vest_count += 1

    # 통계 정보 추가
    cv2.putText(
        result,
        f'Total Vests Detected: {vest_count}',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    return image, result, vest_count


def detect_from_webcam():
    """
    웹캠에서 실시간으로 안전조끼를 검출합니다.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # HSV 변환
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 주황색 마스크
        lower_orange = np.array([5, 100, 100])
        upper_orange = np.array([15, 255, 255])
        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        # 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 컨투어 검출
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 검출 결과 표시
        vest_count = 0
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            vest_count += 1

        # 정보 표시
        cv2.putText(
            frame,
            f'Safety Vests: {vest_count}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # 화면 표시
        cv2.imshow('Original', frame)
        cv2.imshow('Mask', mask)

        # 종료 조건
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    """
    메인 함수
    """
    print("=== Safety Vest Detector ===")
    print("1. Detect from image")
    print("2. Detect from webcam")

    choice = input("Select mode (1 or 2): ")

    if choice == '1':
        image_path = input("Enter image path: ")
        original, result, count = detect_safety_vest(image_path)

        if result is not None:
            print(f"Detected {count} safety vest(s)")

            # 결과 표시
            cv2.imshow('Original', original)
            cv2.imshow('Detection Result', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    elif choice == '2':
        detect_from_webcam()

    else:
        print("Invalid choice")


if __name__ == '__main__':
    main()
