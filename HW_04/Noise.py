import cv2
import numpy as np
import os


# 가우시안 노이즈 생성 함수


def add_gaussian_noise(image, mean=0, std=20):
    noise = np.zeros(image.shape, np.int16)
    # randn 함수로 노이즈 채움
    cv2.randn(noise, mean, std)
    
    # 원본 이미지(uint8)와 노이즈(int16)를 더하기 위해 원본을 int16으로 변환 후 덧셈 수행
    noisy_image = image.astype(np.int16) + noise
    
    #0 미만은 0, 255 초과는 255로 설정
    noisy_image = np.clip(noisy_image, 0, 255)
    
    # 최종 타입을 uint8로 변환하여 반환
    return noisy_image.astype(np.uint8)

# Salt & Pepper 노이즈 생성 함수
def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    """
    salt_prob: 255 (Salt)가 될 확률
    pepper_prob: 0 (Pepper)이 될 확률
    """
    # 원본 이미지를 복사하여 노이즈 이미지를 생성
    noisy_image = np.copy(image)
    
    # 0과 1 사이의 랜덤 실수 행렬을 이미지와 같은 크기로 생성
    random_matrix = np.random.rand(*image.shape)
    
    # Pepper 노이즈 추가 (0)
    # 랜덤 값이 pepper_prob보다 작으면 해당 픽셀을 0으로 설정
    noisy_image[random_matrix < pepper_prob] = 0
    
    # Salt 노이즈 추가 (255)
    # 랜덤 값이 (1.0 - salt_prob)보다 크면 해당 픽셀을 255로 설정
    noisy_image[random_matrix > (1.0 - salt_prob)] = 255
    
    return noisy_image

# mean 필터 구현 함수 

def mean_filter_3x3(image):
    # 이미지 높이, 너비
    height, width = image.shape
    # 결과 이미지 (0으로 초기화)
    output = np.zeros_like(image)
    
    # 경계 처리를 위해 1픽셀 패딩 (경계 값을 복제)
    padded_image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    
    # 모든 픽셀을 순회 (패딩된 이미지가 아닌 원본 이미지 기준)
    for y in range(height):
        for x in range(width):
            # 3x3 윈도우 (영역) 추출
            # (y, x)는 원본 기준이므로 패딩된 이미지에서는 (y+1, x+1)이 중심
            window = padded_image[y:y+3, x:x+3]
            # 윈도우의 평균값 계산
            mean_val = np.mean(window)
            # 결과 이미지에 할당
            output[y, x] = mean_val
            
    return output.astype(np.uint8)

# median 필터 구현 함수 
def median_filter_3x3(image):
    
    height, width = image.shape
    output = np.zeros_like(image)
    
    # 경계 처리를 위해 1픽셀 패딩
    padded_image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    
    # 모든 픽셀을 순회
    for y in range(height):
        for x in range(width):
            # 3x3 윈도우 추출
            window = padded_image[y:y+3, x:x+3]
            # 윈도우의 중앙값(median) 계산
            median_val = np.median(window)
            # 결과 이미지에 할당
            output[y, x] = median_val
            
    return output.astype(np.uint8)

# -------------------------------------------------------------------
# 3. 메인 실행 로직
# -------------------------------------------------------------------

def main():
    # 결과물을 저장할 폴더 생성
    output_dir = "noise_filter_results"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 원본 이미지 로드
    image_path = 'Lena512.jpg'
    # 이미지 흑백처리
    original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 2. 4개의 노이즈 영상 생성
    # 슬라이드: Gaussian 노이즈 2개 (노이즈가 작은 것, 큰 것)
    # -> 표준편차(std) 값으로 조절 (예: 20, 50)
    print("Gaussian 노이즈 영상 2개 생성 중...")
    gauss_small = add_gaussian_noise(original_img, mean=0, std=20)
    gauss_large = add_gaussian_noise(original_img, mean=0, std=50)

    # 슬라이드: Salt&Pepper 노이즈 2개
    # -> 1: noise 확률 각각 0.05 (salt=0.05, pepper=0.05)
    # -> 2: noise 확률 각각 0.1  (salt=0.1, pepper=0.1)
    print("Salt & Pepper 노이즈 영상 2개 생성 중...")
    sp_small = add_salt_and_pepper_noise(original_img, salt_prob=0.05, pepper_prob=0.05)
    sp_large = add_salt_and_pepper_noise(original_img, salt_prob=0.1, pepper_prob=0.1)

    # 생성된 노이즈 영상들
    noise_images = {
        "gauss_small": gauss_small,
        "gauss_large": gauss_large,
        "sp_small": sp_small,
        "sp_large": sp_large
    }
    
    # 원본 및 노이즈 영상 저장
    cv2.imwrite(os.path.join(output_dir, "0_original.png"), original_img)
    for name, img in noise_images.items():
        cv2.imwrite(os.path.join(output_dir, f"1_noisy_{name}.png"), img)

    # 3. 2개의 필터 정의
    filters = {
        "mean_filter": mean_filter_3x3,
        "median_filter": median_filter_3x3
    }

    # 4. 4개의 노이즈 영상 x 2개의 필터조합 실행
    print("\n총 8개의 필터링 조합 실행 중 ...")
    
    result_count = 1
    for noise_name, noisy_img in noise_images.items():
        for filter_name, filter_func in filters.items():
            
            print(f"  {result_count}/8: [{noise_name}] 이미지에 [{filter_name}] 적용 중...")
            
            # 필터 적용
            filtered_image = filter_func(noisy_img)
            
            # 결과 저장
            result_filename = f"2_result_{noise_name}_filtered_by_{filter_name}.png"
            cv2.imwrite(os.path.join(output_dir, result_filename), filtered_image)
            
            result_count += 1

   
    print("(원본 1개, 노이즈 4개, 필터링 결과 8개)")

if __name__ == "__main__":
    main()