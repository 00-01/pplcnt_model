import cv2
import numpy as np
from PIL import Image


img1 = f"../model/000.png"
img2 = f"../model/111.png"

im1 = Image.open(img1)
im2 = Image.open(img2)
img1_arr = np.array(im1, int)
img2_arr = np.array(im2, int)

h, w = 80, 80
min, mid, max = 0, 128, 255


def crop_img(arr):
    new_arr = arr
    lr, tb = w//4, h//4
    # ## side cut
    # new_arr[:, :lr] = 0
    # new_arr[:, w-lr:] = 0
    # ## top cut
    # new_arr[:tb, :] = 0

    new_arr = new_arr[: , w//4:w-w//4]
    new_arr = new_arr[h//4: , :]

    return new_arr

# img3 = np.round_(255-((img1_arr-img2_arr)+255)/2).astype(np.uint8)
img_dist = img1_arr-img2_arr
# img_dist = crop_img(img_dist)
img_norm = np.round_(255-(img_dist+255))    # inverse
img_norm[img_norm < 20] = 0    # low cut
result = img_norm.astype(np.uint8)   # dtype to uint8

window_name = "out"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.imshow(window_name, result)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()




## output = xc, yc