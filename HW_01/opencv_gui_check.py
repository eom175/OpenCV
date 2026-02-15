# opencv_gui_check.py
import cv2, numpy as np
print("cv2:", cv2.__version__)

img = np.linspace(0,255,256, dtype=np.uint8)
img = np.tile(img, (256,1))

cv2.imshow("GUI check", img)
print("Press any key on the image window...")
cv2.waitKey(0)
cv2.destroyAllWindows()