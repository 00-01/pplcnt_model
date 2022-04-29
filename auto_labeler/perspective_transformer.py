import cv2
import numpy as np
from PIL import Image


# w_re, h_re = 80, 80
# # scale = 1200//w_re
# scale = 5
# reshaped_size = (w_re*scale, h_re*scale)
#
# device_id = "00"
# id = "1650497896553"
# datapath_ir = f"/media/z/e9503728-f419-4a14-9fc0-21e2947af50c/DOWNLOAD/{id}.png"
# datapath_rgb = f"/media/z/e9503728-f419-4a14-9fc0-21e2947af50c/DOWNLOAD/{id}.jpg"
# # ir_img = cv2.imread(datapath_ir, cv2.IMREAD_GRAYSCALE)
# ir_img = cv2.imread(datapath_ir)
# rgb_img = cv2.imread(datapath_rgb)
# # rgb_img = cv2.imread(datapath_rgb, cv2.IMREAD_GRAYSCALE)
# ir_img = cv2.resize(ir_img, reshaped_size)
# # rgb_img = cv2.resize(rgb_img, reshaped_size)
#
# array = np.zeros((4,2), int)
# corner = 4
# def click(event, x, y, flag, param):
#     global corner
#     if event == cv2.EVENT_LBUTTONDOWN:
#         cv2.circle(rgb_img, (x,y), 5, (0,0,255), -1)
#         array[corner] = x,y
#         corner += 1
#         print(x, y)
#
#
# def edger(img, thresh1=100, thresh2=200):
#     edge = cv2.Canny(image=img, threshold1=thresh1, threshold2=thresh2)
#     return edge
#
#
# # def overlayer(img1, img2):
# #     alpha = 1.0
# #     while True:
# #         overlayed = cv2.addWeighted(src1=img1, alpha=alpha, src2=img2, beta=1-alpha, gamma=0)
# #         cv2.putText(overlayed, f'alpha: {alpha:.2f}', org=(50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
# #                     fontScale=1, color=(0,0,255), thickness=2,) # lineType=cv2.LINE_AA)
# #         cv2.imshow("overlayed", overlayed)
# #
# #         key = cv2.waitKey()
# #         # print(key)
# #     return key
#
#
# if device_id == "01": x, y, xw, yh = -1.53*scale, -6.73*scale, 1.16, 1.16
# elif device_id == "02": x, y, xw, yh = -8.93*scale, -8.6*scale, 1.16, 1.16
# elif device_id == "03": x, y, xw, yh = -2.4*scale, -8.53*scale, 1.16, 1.16
# elif device_id == "05": x, y, xw, yh = -2.4*scale, -8.53*scale, 1.16, 1.16
# elif device_id == "07": x, y, xw, yh = -8.2*scale, -7.6*scale, 1.16, 1.16
# else: x, y, xw, yh = 0, 0, 1, 1
# # if device_id == "01": x, y, w, h = -23/scale, -101/scale, 1.16, 1.16
# # elif device_id == "02": x, y, w, h = -134/scale, -129/scale, 1.16, 1.16
# # elif device_id == "03": x, y, w, h = -36/scale, -128/scale, 1.16, 1.16
# # else: x, y, xw, yh = 1, 1, 0, 0
#
# alpha = 1.0
# while True:
#     # cv2.namedWindow('Input')
#     # cv2.setMouseCallback('Input', click)
#     # cv2.imshow('Input', rgb_img)
#     # key = cv2.waitKey(1)
#     # print(key)
#
#     # if corner == 4:
#     #     pts1 = np.float32([[array[0], array[1], array[2], array[3]]])
#     #     pts2 = np.float32([[0,0], [w_re,0], [0,h_re], [w_re,h_re]])
#
#         # matrix = cv2.getPerspectiveTransform(pts1, pts2)
#     matrix = np.float32([[xw, 0, y],
#                          [0, yh, x],
#                          [0, 0, 1]])
#
#     warped = cv2.warpPerspective(rgb_img, matrix, reshaped_size)
#     # warped = cv2.imread("warped.jpg")
#
#     # edge2 = edger(warped, 20, 20)
#     # edge1 = edger(ir_img, 200, 200)
#     # edge1 = cv2.resize(edge1, reshaped_size)
#     # overlayed = cv2.addWeighted(src1=edge1, alpha=alpha, src2=edge2, beta=1-alpha, gamma=0)
#
#     overlayed = cv2.addWeighted(src1=ir_img, alpha=alpha, src2=warped, beta=1-alpha, gamma=0)
#     # cv2.putText(overlayed, f'alpha:{alpha:.2f}, x:{x}, y:{y}, w:{w:.2f}, h:{h:.2f}', org=(50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=1,) # lineType=cv2.LINE_AA)
#     cv2.imshow("overlayed", overlayed)
#
#     key = cv2.waitKey()
#     # up: 82, down:84, left: 81, right: 83
#     if key == ord('q'):
#         alpha -= 0.02
#         if alpha < 0: alpha = 0
#     elif key == ord('w'):
#         alpha += 0.02
#         if alpha > 1: alpha = 1
#
#     elif key == 82: x -= 1
#     elif key == 84: x += 1
#
#     elif key == 81: y -= 1
#     elif key == 83: y += 1
#
#     elif key == ord('a'): xw -= 0.01
#     elif key == ord('s'): xw += 0.01
#
#     elif key == ord('z'): yh -= 0.01
#     elif key == ord('x'): yh += 0.01
#
#     elif key == ord('p'): cv2.imwrite("warped.jpg", warped)
#
#     elif key == 27:
#         cv2.destroyAllWindows()
#         break
#
#     print(f'{x}, {y}, {xw:.2f}, {yh:.2f}')

base_dir = "/home/z/MVPC10/CODE/pplcnt_model/data/"

data_prev = "20220428-075507_03"
data_now = "20220428-075608_03"

data_prev_ir = f"{base_dir}{data_prev}_IR.png"
data_prev_rgb = f"{base_dir}{data_prev}_RGB.jpg"
data_now_ir = f"{base_dir}{data_now}_IR.png"
data_now_rgb = f"{base_dir}{data_now}_RGB.jpg"

files = []
files.append(data_prev_ir)
files.append(data_prev_rgb)
files.append(data_now_ir)
files.append(data_now_rgb)


def tranformer(rgb, ir):
    w_re, h_re = ir.shape[:2]
    # scale = 1200//w_re
    scale = 5
    reshaped_size = (w_re*scale, h_re*scale)

    rgb_reshaped = cv2.resize(rgb, reshaped_size)
    ir_reshaped = cv2.resize(ir, reshaped_size)

    device_id = "03"

    if device_id == "01": x, y, xw, yh = -1.53*scale, -6.73*scale, 1.16, 1.16
    elif device_id == "02": x, y, xw, yh = -8.93*scale, -8.6*scale, 1.16, 1.16
    elif device_id == "03": x, y, xw, yh = -2.4*scale, -8.53*scale, 1.16, 1.16
    elif device_id == "05": x, y, xw, yh = -2.4*scale, -8.53*scale, 1.16, 1.16
    elif device_id == "07": x, y, xw, yh = -8.2*scale, -7.6*scale, 1.16, 1.16
    else: x, y, xw, yh = 0, 0, 1, 1

    alpha = 1.0
    while True:
        matrix = np.float32([[xw, 0, y], [0, yh, x], [0, 0, 1]])
        warped = cv2.warpPerspective(rgb_reshaped, matrix, reshaped_size)
        print(warped.shape, ir_reshaped.shape)
        overlayed = cv2.addWeighted(src1=ir_reshaped, alpha=alpha, src2=warped, beta=1-alpha, gamma=0)
        # cv2.putText(overlayed, f'alpha:{alpha:.2f}, x:{x}, y:{y}, w:{w:.2f}, h:{h:.2f}', org=(50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=1,) # lineType=cv2.LINE_AA)
        cv2.imshow("overlayed", overlayed)

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

        elif key == ord('p'): cv2.imwrite("warped.jpg", warped)

        elif key == 27:
            cv2.destroyAllWindows()
            break

        print(f'{x}, {y}, {xw:.2f}, {yh:.2f}')


ir = cv2.imread(data_prev_ir)
rgb = cv2.imread(data_prev_rgb)

tranformer(rgb, ir)