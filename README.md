# OpenCV Computer Vision Practice

컴퓨터 비전 수업 실습 코드를 주차별(`HW_01` ~ `HW_05`)로 정리한 저장소입니다.

## Environment

- Python 3.9+
- `opencv-python`
- `numpy`
- `matplotlib` (HW_05 시각화에 사용)

설치:

```bash
pip install opencv-python numpy matplotlib
```

## Folder Structure

- `HW_01`
  - `hw_1.py`: Grayscale bilinear resize 구현
  - `hw2.py`: Forward mapping 기반 영상 회전 실습
  - `opencv_gui_check.py`: OpenCV GUI 동작 확인
- `HW_02`
  - `Negative_transform.py`: Negative transform 구현
  - `gamma_transform.py`: Gamma correction 구현
  - `HistogramEqualization.py`: 수동 Histogram Equalization 구현
- `HW_03`
  - `filtering.py`: 사용자 정의 커널 기반 spatial filtering 실습
- `HW_04`
  - `Noise.py`: Gaussian / Salt&Pepper 노이즈 생성 및 Mean/Median filtering
- `HW_05`
  - `EdgeDetection.py`: Sobel 기반 edge detection 및 smoothing 비교

## Run

각 주차 폴더로 이동해서 스크립트를 실행하면 됩니다.

```bash
cd HW_03
python filtering.py
```

참고:

- 일부 스크립트는 `cv2.imshow()`를 사용하므로 GUI 환경에서 실행해야 합니다.
- 입력 이미지는 각 `HW_*` 폴더에 포함되어 있습니다.
