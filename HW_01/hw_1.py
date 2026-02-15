

import cv2
import numpy as np


def bilinear_resize_gray(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:

    h, w = img.shape
    scale_x = w / new_w
    scale_y = h / new_h

    out = np.zeros((new_h, new_w), dtype=np.uint8)

    for y in range(new_h):
        for x in range(new_w):
            gx = x * scale_x
            gy = y * scale_y

            x1 = int(np.floor(gx))
            y1 = int(np.floor(gy))
            x2 = min(x1 + 1, w - 1)
            y2 = min(y1 + 1, h - 1)

            Q11 = img[y1, x1]
            Q21 = img[y1, x2]
            Q12 = img[y2, x1]
            Q22 = img[y2, x2]

            dx = gx - x1
            dy = gy - y1

            val = (Q11 * (1 - dx) * (1 - dy) +
                   Q21 * dx       * (1 - dy) +
                   Q12 * (1 - dx) * dy       +
                   Q22 * dx       * dy)

            out[y, x] = int(val)

    return out

if __name__ == "__main__":
    
    img = cv2.imread("Lena_256x256.png", cv2.IMREAD_GRAYSCALE)
    resized_img = bilinear_resize_gray(img, 512, 512)
    cv2.imshow("Lena_512x512_bilinear.png", resized_img)

    img2 = cv2.imread("Lena_256x256.png", cv2.IMREAD_GRAYSCALE)
    resized_img2 = bilinear_resize_gray(img2, 436, 436)
    cv2.imshow("Lena_436x436_bilinear.png", resized_img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
