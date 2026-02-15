import cv2
import numpy as np

# 1. 이미지 파일을 흑백(Grayscale)으로 불러오기

img_gray = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
 

# 2. Negative Transformation을 픽셀 단위로 직접 구현
height, width = img_gray.shape
negative_img = np.zeros((height, width), dtype=np.uint8)

# 모든 픽셀을 순회하며 네거티브 변환 적용
for y in range(height):
    for x in range(width):
        # 원본 픽셀 값(r)
        r = img_gray[y, x]
        # 네거티브 변환 공식(s = 255 - r) 적용
        s = 255 - r
        # 결과 이미지에 값 할당
        negative_img[y, x] = s

# 3. 결과 이미지 파일로 저장
cv2.imwrite('Lena_negative.png', negative_img)

# 4. 원본과 결과 이미지 화면에 출력
cv2.imshow('Original Grayscale', img_gray)
cv2.imshow('Negative Transformation', negative_img)

cv2.waitKey(0)
cv2.destroyAllWindows()