import cv2
import numpy as np


def custom_filtering(image, n, kernel):
    # 입력 이미지의 높이, 너비, 채널 수 확인
    is_color = len(image.shape) == 3
    if is_color: # 맞다면 각각에 대입
        height, width, channels = image.shape
    else: # 흑백이미지의 경우에도 처리할 수 있도록 채널 추가
        height, width = image.shape
        image = np.expand_dims(image, axis=2)
        height, width, channels = image.shape
    #제로 패딩으로 구현() ->> 원본 크기 유지하면서 커널을 적용하기 위함
    padded_height = height + 2 * n
    padded_width = width + 2 * n
    #0으로 채워진 패딩 이미지 생성
    padded_image = np.zeros((padded_height, padded_width, channels), dtype=np.uint8)

    #원본 이미지를 패딩 이미지 중앙에 복사
    padded_image[n:n + height, n:n + width] = image
    #출력 이미지 초기화
    output_image = np.zeros_like(image, dtype=np.float64)
    #각 채널에 대해 가로, 세로 픽셀을 순회하면서 필터링 적용
    for c in range(channels):
        for y in range(height):
            for x in range(width):
                region = padded_image[y:y + 2*n + 1, x:x + 2*n + 1, c]
                pixel_value = np.sum(region * kernel) # Convolution 연산
                output_image[y, x, c] = pixel_value

    #출력 이미지의 픽셀 값을 0~255 범위로 제한하고, uint8 타입으로 변환
    output_image = np.clip(output_image, 0, 255)
    output_image = output_image.astype(np.uint8)
    if not is_color:
        output_image = output_image.squeeze()
    return output_image
# =================================================================


if __name__ == '__main__':
    
    try:
        image = cv2.imread('Lena.png', cv2.IMREAD_COLOR) #칼라 이미지로 읽기
        if image is None:
            raise FileNotFoundError("Lena.png 파일을 찾을 수 없습니다.")

        # --- 1. Moving Average 필터 (다양한 크기) ---
        print("Moving Average 필터를 적용합니다...")
        
        # 3x3 Moving Average
        n_3 = 1
        kernel_avg_3x3 = np.ones((3, 3)) / 9.0
        result_avg_3x3 = custom_filtering(image, n_3, kernel_avg_3x3)

        # 5x5 Moving Average
        n_5 = 2
        kernel_avg_5x5 = np.ones((5, 5)) / 25.0
        result_avg_5x5 = custom_filtering(image, n_5, kernel_avg_5x5)

        # 7x7 Moving Average
        n_7 = 3
        kernel_avg_7x7 = np.ones((7, 7)) / 49.0
        result_avg_7x7 = custom_filtering(image, n_7, kernel_avg_7x7)

        # 13x13 Moving Average
        n_13 = 6
        kernel_avg_13x13 = np.ones((13, 13)) / 169.0
        result_avg_13x13 = custom_filtering(image, n_13, kernel_avg_13x13)
        
        # --- 2. Laplacian 필터 (3x3) ---
        print("Laplacian 필터를 적용합니다...")
        n_lap = 1
        # 라플라시안 필터는 2차 미분을 이용해 경계를 검출
        kernel_laplacian = np.array([
            [0,  1, 0],
            [1, -4, 1],
            [0,  1, 0]
        ])
        result_laplacian = custom_filtering(image, n_lap, kernel_laplacian)

        # --- 3. Sharpening 필터 (Laplacian 기반) ---
        print("Sharpening 필터를 적용합니다 (원본 - 라플라시안 방식)...")
        
        # 원본 이미지와 라플라시안 결과 모두 float32로 변환하여 연산 수행
        image_float = image.astype(np.float32)
        laplacian_float = result_laplacian.astype(np.float32)
        
        # Sharpening 연산: 원본 - 라플라시안 
        sharpened_image_float = image_float - laplacian_float
        
   
        result_sharpen_subtraction = np.clip(sharpened_image_float, 0, 255).astype(np.uint8)




        # --- 결과 이미지 출력 ---
        cv2.imshow('Original', image)
        
        # Moving Average 결과
        cv2.imshow('Moving Average 3x3', result_avg_3x3)
        cv2.imshow('Moving Average 7x7', result_avg_7x7)
        cv2.imshow('Moving Average 13x13', result_avg_13x13)
        
        # Laplacian & Sharpening 결과
        cv2.imshow('Laplacian (Edge Detection)', result_laplacian)
        cv2.imshow('Sharpened', result_sharpen_subtraction)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")