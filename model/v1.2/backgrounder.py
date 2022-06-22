from glob import glob

import cv2
import numpy as np
import pandas as pd
from PIL import Image


img_list = glob(f"test_data/*", recursive=0)
outpath = f"."
csvfile = f"{outpath}/v1.2_train.csv"
df = pd.read_csv(f'{csvfile}')

h, w = 80, 80
min, mid, max = 0, 128, 255
error = 239
back_img_list = []
ppl_thresh = 1


def crop_img(arr):
    new_arr = arr
    # lr, tb = w//4, h//4
    new_arr = new_arr[:, w//4:w-w//4]
    new_arr = new_arr[h//4:, :]

    return new_arr


def bg_remover(target, bg_list):
    bg = np.zeros([80, 80], dtype=int)
    for i in bg_list:
        img = Image.open(i)
        img_arr = np.array(img, int)
        bg += img_arr
    bg //= len(bg_list)

    img = target-bg
    # img = crop_img(img_dist)
    img = np.round_(255-(img+255))  # inverse
    img[img < 15] = 0  # low cut
    result = img.astype(np.uint8)  # dtype to uint8

    return result


backgrounds = []
for i in range(len(df)):
    bg_len = len(backgrounds)
    img_path = df.iloc[i, 0]
    img_cnt = df.iloc[i, 1]

    img = Image.open(img_path)
    img_arr = np.array(img, int)
    error = img_arr[img_arr > 239]
    if len(error) > 512:  continue  ## filter error img

    if bg_len == 5:
        result = bg_remover(img_arr, backgrounds)
        '''DO MODEL'''
        if img_cnt < ppl_thresh:
            backgrounds.insert(0, img_path)
            backgrounds.pop(-1)
    else:
        if img_cnt < ppl_thresh:
            backgrounds.insert(0, img_path)
        continue

    window_name = "out"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.putText(result, str(img_cnt), (35, 5), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1)
    cv2.imshow(window_name, result)
    ## print(i)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()
        break
    elif key == ord('s'):
        cv2.imwrite('result.png', result)

# result = Image.fromarray(result)
# result = result.resize((400, 600))
# display(result)
