from glob import glob
import os

import cv2
import numpy as np
from PIL import Image


base = "/media/z/0/MVPC10/DATA"
device = "device_03"
path = f"{base}/{device}"
mask_output = "output/mask"
mask_path = f"{path}/{mask_output}"

ir_dir = f"{path}/**/*.png"
mask_dir = f"{mask_path}/**/*.jpg"

ir_paths = glob(f'{ir_dir}', recursive=False)
mask_paths = glob(f'{mask_dir}', recursive=False)


def tranformer(layer, base, scale):
    h_re, w_re = base.shape[:2]
    reshaped_size = (h_re*scale, w_re*scale)

    layer_reshaped = cv2.resize(layer, reshaped_size)
    base_reshaped = cv2.resize(base, reshaped_size)

    # device_id = "03"
    # if device_id == "01": x, y, xw, yh = -1.53*scale, -6.73*scale, 1.16, 1.16
    # elif device_id == "02": x, y, xw, yh = -8.93*scale, -8.6*scale, 1.16, 1.16
    # elif device_id == "03": x, y, xw, yh = -1.5*scale, -11.8*scale, 1.25, 1.12
    # elif device_id == "05": x, y, xw, yh = -2.4*scale, -8.53*scale, 1.16, 1.16
    # elif device_id == "07": x, y, xw, yh = -8.2*scale, -7.6*scale, 1.16, 1.16
    # else: x, y, xw, yh = 0, 0, 1.16, 1.16
    # x, y, xw, yh = 0, 0, 1, 1
    # x, y, xw, yh = -1.7*scale, -9.9*scale, 1.21, 1.12
    # x, y, xw, yh = -1.5*scale, -8.9*scale, 1.18, 1.12
    # x, y, xw, yh = -2.8*scale, -8.9*scale, 1.18, 1.15
    x, y, xw, yh = -2.94*scale, -7.56*scale, 1.15, 1.15

    alpha = 0.5
    while True:
        matrix = np.float32([[xw, 0, y], [0, yh, x], [0, 0, 1]])
        warped = cv2.warpPerspective(layer_reshaped, matrix, reshaped_size)
        overlayed = cv2.addWeighted(src1=base_reshaped, alpha=alpha, src2=warped, beta=1-alpha, gamma=0)
        # cv2.putText(overlayed, f'alpha:{alpha:.2f}, x:{x}, y:{y}, w:{w:.2f}, h:{h:.2f}', org=(50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=1,) # lineType=cv2.LINE_AA)
        win_name = "label"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, overlayed)

        key = cv2.waitKey()
        # up: 82, down:84, left: 81, right: 83
        if key == ord('q'):
            alpha -= 0.02
            if alpha < 0: alpha = 0
        elif key == ord('w'):
            alpha += 0.02
            if alpha > 1: alpha = 1

        elif key == 82: x -= 1
        elif key == 84: x += 1
        elif key == 81: y -= 1
        elif key == 83: y += 1

        elif key == ord('a'): xw -= 0.01
        elif key == ord('s'): xw += 0.01
        elif key == ord('z'): yh -= 0.01
        elif key == ord('x'): yh += 0.01

        # elif key == ord('p'): cv2.imwrite("warped.jpg", warped)
        # elif key == ord('p'): break

        elif key == 27:
            # cv2.destroyAllWindows()
            break
        print(f'{x:.2f}, {y:.2f}, {xw:.2f}, {yh:.2f}')


def masker():
    ir_list = []
    for i in mask_paths:
        # print(i, j)
        dir_num = os.path.basename(os.path.dirname(i))
        img_num = os.path.basename(i).replace(".jpg", ".png")
        # print(dir_num, img_num)

        ir_list.append(f"{path}/{dir_num}/{img_num}")

    return ir_list


# data_num = "1652648716705"
# data_num = "1652657356818"
# data_num = "0"
# align_img(data_num)


ir_list = masker()
# print(mask_list)
temp = f"data/temp"
for i, j in zip(ir_list, mask_paths):
    ir = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(j, cv2.IMREAD_GRAYSCALE)
    h, w = mask.shape[0], mask.shape[1]
    mask_arr = np.array(mask)
    split_mask = np.split(mask_arr, h/w, axis=0)
    for i, j in enumerate(split_mask):
        im = Image.fromarray(np.uint8(j))
        mask_save = f"{temp}/{i}.png"
        im.save(mask_save)
        mask_saved = cv2.imread(mask_save, cv2.IMREAD_GRAYSCALE)

        tranformer(mask_saved, ir, 1)

        # if key == 27:
        #     cv2.destroyAllWindows()
        #     break
