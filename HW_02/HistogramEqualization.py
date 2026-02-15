import cv2
import numpy as np
img_gray = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)


# Step #1: Lena 이미지 변환
# s = r/2 변환을 적용하여 'lena1' 이미지 생성 (전체적으로 어두워짐).
lena1 = (img_gray / 2).astype(np.uint8)

# s = 128 + r/2 변환을 적용하여 'lena2' 이미지 생성 (전체적으로 밝아지고 대비가 감소).
lena2 = (128 + img_gray / 2).astype(np.uint8)


# Histogram Equalization 구현 함수
def manual_equalize_hist(image):
    """
    강의에 나오는 공식 s = T(r) = (L-1) * ∫[0 to r] p_r(w)dw 를 코드로 구현
    """
    height, width = image.shape

    # --- 1단계: Histogram 계산 ---
    # 각 밝기 값(0~255)에 해당하는 픽셀의 개수를 셈.
    # 강의에서 설명한 확률 밀도 함수 p_r(w)에 해당
    hist = np.zeros(256, dtype=int)
    for y in range(height):
        for x in range(width):
            hist[image[y, x]] += 1
            
    # --- 2단계: 누적 분포 함수(CDF) 계산 ---
    # Histogram의 누적 합계 계산. cdf[i]는 밝기 i까지의 모든 픽셀 수를 의미.
    # 강의 슬라이드에서 나오는 적분 부분인 ∫[0 to r] p_r(w)dw (누적 분포 함수)를 계산하는 과정.
    cdf = np.zeros(256, dtype=int)
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + hist[i]

    # --- 3단계: 변환 맵(Look-Up Table) 생성 ---
    # 위에서 계산한 CDF를 사용하여 원본 픽셀 값을 새로운 값으로 매핑하는 표 생성
    
    # 대비를 효과적으로 늘리기 위해, CDF의 최소값(0이 아닌 첫 번째 값)을 찾음.
    cdf_min = cdf[np.nonzero(cdf)[0][0]]
    total_pixels = height * width  # 이미지의 전체 픽셀 수

    transform_map = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        # Equalization 공식을 적용하여 각 밝기(i)가 변환될 새로운 값(s)을 계산.
        # (cdf[i] - cdf_min) / (total_pixels - cdf_min) : 누적 분포(CDF)를 0~1 사이로 정규화하는 과정. (∫p_r(w)dw 에 해당)
        # * 255 : 정규화된 값을 다시 0~255 범위로 스케일링. ((L-1)에 해당)
        numerator = cdf[i] - cdf_min
        denominator = total_pixels - cdf_min
        transform_map[i] = round((numerator / denominator) * 255)

    # --- 4단계: 변환 맵을 사용하여 새로운 이미지 생성 ---
    # 원본 이미지의 모든 픽셀 값을 transform_map을 참조하여 새로운 값으로 변경.
    equalized_img = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            original_pixel = image[y, x]
            equalized_img[y, x] = transform_map[original_pixel]
            
    return equalized_img

# ------------------------------------------------------------------
# Step #2: 각 이미지에 대해 Histogram Equalization 함수 적용
# ------------------------------------------------------------------
eq_original = manual_equalize_hist(img_gray)
eq_lena1 = manual_equalize_hist(lena1)
eq_lena2 = manual_equalize_hist(lena2)

# ------------------------------------------------------------------
# 결과 이미지들을 화면에 출력
# ------------------------------------------------------------------
cv2.imshow('Original Grayscale', img_gray)
cv2.imshow('Equalized Original', eq_original)
cv2.imshow('lena1 (Dark)', lena1)
cv2.imshow('Equalized lena1', eq_lena1)
cv2.imshow('lena2 (Bright)', lena2)
cv2.imshow('Equalized lena2', eq_lena2)

cv2.waitKey(0)
cv2.destroyAllWindows()