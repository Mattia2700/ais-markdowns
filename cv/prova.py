import cv2
import numpy as np

image = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 255, 0, 255, 0, 0, 0, 0, 0],
                  [0, 0, 0, 255, 255, 255, 255, 255, 0],
                  [0, 255, 255, 0, 0, 255, 0, 0, 0],
                  [0, 0, 255, 0, 255, 0, 0, 0, 0],
                  [0, 255, 255, 255, 0, 255, 255, 0, 0],
                  [0, 0, 0, 255, 255, 255, 0, 0, 0],
                  [0, 0, 0, 255, 0, 255, 0, 0, 0]], dtype=np.uint8)

kernel = np.ones((3, 2), dtype=np.uint8) * 255

eroded = cv2.erode(image, kernel, anchor=(0, 1), iterations=1)

# open image
cv2.imshow('image', eroded)
cv2.waitKey(0)