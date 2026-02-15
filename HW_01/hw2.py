import cv2
import numpy as np

def rotate_center_forward_same_size(img, angle_deg, bg=0):
    """
    슬라이드 수식: [x'; y'] = [[cos -sin],[sin cos]] [x; y]
    정방향 회전(Forward mapping). 구멍 방지를 위해 4-이웃 가중치 분배(splatting) 적용.
    """


    h, w = img.shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    th = np.deg2rad(angle_deg)
    c, s = np.cos(th), np.sin(th)

    R = np.array([[c, -s],
                  [s,  c]], dtype=np.float64)

    
    acc   = np.zeros((h, w), dtype=np.float64)
    wsum  = np.zeros((h, w), dtype=np.float64)

    
    for ys in range(h):
        for xs in range(w):
            
            xr, yr = np.array([xs - cx, ys - cy]) @ R + np.array([cx, cy])

            
            x1 = int(np.floor(xr)); y1 = int(np.floor(yr))
            dx = xr - x1;          dy = yr - y1

            x2 = x1 + 1; y2 = y1 + 1

            
            nbrs = [
                (x1, y1, (1 - dx) * (1 - dy)),
                (x2, y1,      dx   * (1 - dy)),
                (x1, y2, (1 - dx) *      dy ),
                (x2, y2,      dx   *      dy ),
            ]

            val = float(img[ys, xs])

            
            for xq, yq, wgt in nbrs:
                if 0 <= xq < w and 0 <= yq < h and wgt > 0.0:
                    acc[yq, xq]  += val * wgt
                    wsum[yq, xq] += wgt

   
    out = np.full((h, w), float(bg), dtype=np.float64)
    mask = wsum > 1e-8
    out[mask] = acc[mask] / wsum[mask]

    return np.clip(out, 0, 255).astype(np.uint8)



if __name__ == "__main__":
    # 1) 이미지 읽기 + 그레이 변환
    img_color = cv2.imread("Lena.png")
    if img_color is None:
        raise FileNotFoundError("Lena.png 파일을 찾을 수 없습니다.")
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # 2) 정방향 회전출력 격자(grid)와 정확히 일치하지 않으므로 빈 픽셀(hole)이 생김
    rot30 = rotate_center_forward_same_size(gray, 30, bg=0)
    rot45 = rotate_center_forward_same_size(gray, 45, bg=0)
    rot60 = rotate_center_forward_same_size(gray, 60, bg=0)

    # 3) 결과 표시 
    cv2.imshow("Original (Gray)", gray)
    cv2.imshow("Rotated 30 (forward, bilinear splat, bg=0)", rot30)
    cv2.imshow("Rotated 45 (forward, bilinear splat, bg=0)", rot45)
    cv2.imshow("Rotated 60 (forward, bilinear splat, bg=0)", rot60)

    # 4) 저장
    cv2.imwrite("Lena_rot30_forward.png", rot30)
    cv2.imwrite("Lena_rot45_forward.png", rot45)
    cv2.imwrite("Lena_rot60_forward.png", rot60)

    cv2.waitKey(0)
    cv2.destroyAllWindows()