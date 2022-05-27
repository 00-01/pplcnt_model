import cv2
import numpy as np
from PIL import Image


mask_raw = f"../data/1652649677269.jpg"

temp = f"../data/temp"

# temp = f"../data/temp"

def mask_splitter(inPath, outPath):
    mask = cv2.imread(inPath, cv2.IMREAD_GRAYSCALE)
    h, w = mask.shape[0], mask.shape[1]
    mask_arr = np.array(mask)
    split_mask = np.split(mask_arr, h/w, axis=0)
    for i, j in enumerate(split_mask):
        im = Image.fromarray(np.uint8(j))
        mask_save = f"{outPath}/{i}.png"
        im.save(mask_save)
        # mask_saved = cv2.imread(mask_save, cv2.IMREAD_GRAYSCALE)


def mask_concatter(inPath, outPath):
    for i in inPath:
        mask = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        mask_arr = np.array(mask)


        mask_save = f"{outPath}/{i}.png"
        mask_arr.save(mask_save)


# mask_splitter(mask_raw, temp)

# mask_concatter(temp, temp)


mask = cv2.imread(mask_raw, cv2.IMREAD_GRAYSCALE)
h, w = mask.shape[0], mask.shape[1]
mask_arr = np.array(mask)

reshaped_mask = mask_arr.reshape(3, 400, 400)
new_mask = np.transpose(reshaped_mask, (1,2,0))
new = np.max(reshaped_mask, axis=0)

new[new > 128] = 255
new[new < 128] = 0

im = Image.fromarray(np.uint8(new)).convert('L')
mask_save = f"{temp}/10.png"
im.save(mask_save)
