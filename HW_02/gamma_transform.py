import cv2
import numpy as np


img_gray = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

# 2. 감마 변환을 픽셀 단위로 직접 구현
def gamma_correction_manual(image, gamma):
    # 결과 이미지를 저장할 비어있는 배열 생성
    height, width = image.shape
    output_image = np.zeros((height, width), dtype=np.uint8)

    # 모든 픽셀을 순회 (y: 행, x: 열)
    for y in range(height):
        for x in range(width):
            # 현재 픽셀 값(r)을 가져옴
            pixel_value = image[y, x]
            
            # 감마 변환 공식 적용
            transformed_value = (pixel_value / 255.0) ** gamma * 255.0
            # 결과 값이 0~255 범위를 벗어나지 않도록 처리 후, 정수로 변환
            output_image[y, x] = np.clip(transformed_value, 0, 255).astype(np.uint8)
            
    return output_image

# 3. Gamma 값 0.5 및 1.5 적용
gamma_0_5 = 0.5
gamma_1_5 = 1.5

#함수 호출
gamma_transformed_0_5 = gamma_correction_manual(img_gray, gamma_0_5)
gamma_transformed_1_5 = gamma_correction_manual(img_gray, gamma_1_5)

# 4. 결과 이미지 화면에 출력
cv2.imshow('Original Grayscale', img_gray)
cv2.imshow(f'Gamma: {gamma_0_5}', gamma_transformed_0_5)
cv2.imshow(f'Gamma: {gamma_1_5}', gamma_transformed_1_5)

cv2.waitKey(0)
cv2.destroyAllWindows()