import cv2
import numpy as np
import matplotlib.pyplot as plt # 결과 출력을 위한 라이브러리, 계산에는 사용되지 않음

def my_convolution(image, kernel):
    """
    padding을 0으로 설정하여 입력과 출력의 크기가 같도록 설정
    """
    h, w = image.shape
    k_h, k_w = kernel.shape
    
    # 커널의 중심점 계산 (padding 크기 결정)
    pad_h = k_h // 2
    pad_w = k_w // 2
    
    # 이미지 가장자리 0으로 패딩 
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    # 결과 이미지 초기화 (음수 값 보존 및 계산 정확도를 위해 float64 사용)
    output = np.zeros((h, w), dtype=np.float64)
    
    # 컨볼루션 연산 
    for i in range(h):
        for j in range(w):
            roi = padded_image[i:i+k_h, j:j+k_w]
            output[i, j] = np.sum(roi * kernel)
            
    return output

def my_threshold(image, threshold_value):
    """
    픽셀 값이 threshold_value보다 크면 255, 아니면 0으로 설정
    """
    h, w = image.shape
    # 결과 이미지 생성 (초기값 0)
    binary_image = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
         for j in range(w):
             if image[i, j] > threshold_value:
                 binary_image[i, j] = 255
             else:
                 binary_image[i, j] = 0
    
    return binary_image

def apply_sobel_edge_detection(image, threshold=150):
    # 1. Sobel mask 정의 
    gx_kernel = np.array([[-1, 0, 1], 
                          [-2, 0, 2], 
                          [-1, 0, 1]])
    
    gy_kernel = np.array([[-1, -2, -1], 
                          [0, 0, 0], 
                          [1, 2, 1]])

    # 2. 생성한 마스크로 g_x, g_y 계산
    gx = my_convolution(image, gx_kernel)
    gy = my_convolution(image, gy_kernel)

    # 3. 절대값(|gx|, |gy|) 계산 
    abs_gx = np.absolute(gx)
    abs_gy = np.absolute(gy)

    # 4.(|gx| + |gy|)계산  
    magnitude = abs_gx + abs_gy

    # 5. 시각화를 위해 0~255 범위로 변환
    abs_gx_uint8 = np.uint8(np.clip(abs_gx, 0, 255))
    abs_gy_uint8 = np.uint8(np.clip(abs_gy, 0, 255))
    magnitude_uint8 = np.uint8(np.clip(magnitude, 0, 255))

    #6.Thresholding 적용
    edge_image = my_threshold(magnitude_uint8, threshold)

    return abs_gx_uint8, abs_gy_uint8, magnitude_uint8, edge_image

def add_gaussian_noise(image):
    """
    영상에 Gaussian Noise를 추가하는 함수 
    """
    row, col = image.shape
    mean = 0
    var = 100 # 노이즈 강도
    sigma = var ** 0.5
    
    # randn 함수 활용 
    gauss = np.random.randn(row, col) * sigma + mean
    noisy = image + gauss
    
    # 클리핑 및 형변환
    noisy = np.uint8(np.clip(noisy, 0, 255))
    return noisy

# --- 메인 실행 코드 ---

# 1. 이미지 로드 및 그레이스케일 변환 
img_path = 'Lena512.jpg' 
original_img = cv2.imread(img_path)

if original_img is None:
    print("이미지를 찾을 수 없습니다.")
    # 테스트용 빈 이미지
    original_img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.rectangle(original_img, (100,100), (400,400), (255,255,255), -1)

gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

# 1: Smoothing 없는 결과 (기본 Sobel) ===
# 적절한 Threshold 값 설정 
THRESHOLD_VALUE = 150  
print("기본 Sobel Edge 검출 진행 중...")
gx_img, gy_img, mag_img, edge_no_smooth = apply_sobel_edge_detection(gray_img, THRESHOLD_VALUE)

# 2: Noise 추가에 따른 효과 확인 ===
noisy_img = add_gaussian_noise(gray_img)
print("Noisy 이미지 Edge 검출 진행 중...")
_, _, _, edge_noisy = apply_sobel_edge_detection(noisy_img, THRESHOLD_VALUE)

# 실험 1: 약한 스무딩 (커널 3x3, sigma 1)
smooth_img1 = cv2.GaussianBlur(noisy_img, (3, 3), 1)
print("Smoothing 이미지 Edge 검출 진행 중...")
_, _, _, edge_smooth1 = apply_sobel_edge_detection(smooth_img1, THRESHOLD_VALUE)

# 실험 2: 강한 스무딩 (커널 9x9, sigma 3) [cite: 26, 27]
smooth_img2 = cv2.GaussianBlur(noisy_img, (9, 9), 3)
print("Smoothing(9x9) 이미지 Edge 검출 진행 중...")
_, _, _, edge_smooth2 = apply_sobel_edge_detection(smooth_img2, THRESHOLD_VALUE)


# === 결과 출력 ===
plt.figure(figsize=(12, 10))

# 1. |Gx|, |Gy|, |Gx|+|Gy| 
plt.subplot(3, 3, 1)
plt.title("|Gx|")
plt.imshow(gx_img, cmap='gray')
plt.axis('off')

plt.subplot(3, 3, 2)
plt.title("|Gy|")
plt.imshow(gy_img, cmap='gray')
plt.axis('off')

plt.subplot(3, 3, 3)
plt.title("|Gx| + |Gy|")
plt.imshow(mag_img, cmap='gray')
plt.axis('off')

# 2. Threshold 적용시 결과 (Noise 없음)
plt.subplot(3, 3, 4)
plt.title(f"Original Edge (Threshold={THRESHOLD_VALUE})")
plt.imshow(edge_no_smooth, cmap='gray')
plt.axis('off')

# 3. Noise 적용시 결과
plt.subplot(3, 3, 5)
plt.title("Noisy Image Edge")
plt.imshow(edge_noisy, cmap='gray')
plt.axis('off')

# 4. Smoothing 결과
plt.subplot(3, 3, 7)
plt.title("Smooth (3x3, s=1) Edge")
plt.imshow(edge_smooth1, cmap='gray')
plt.axis('off')

plt.subplot(3, 3, 8)
plt.title("Smooth (9x9, s=3) Edge")
plt.imshow(edge_smooth2, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()