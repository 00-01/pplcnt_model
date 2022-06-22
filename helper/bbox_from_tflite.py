from argparse import ArgumentParser
from glob import glob

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf


SAVE = 0
SHOW = 1

r, g, b = (0,0,255), (0,255,0), (255,0,0)

# parser = ArgumentParser()
# parser.add_argument("-s", "--show", default=0, type=int, help="show image")
# args = parser.parse_args()

img_path = f"input"
img_list = glob(f"{img_path}/*", recursive=0)
# img_list = ["../model/0101.png"]
model_path = "../model/v1.0/v1.0.1.tflite"

min, max = 0, 255
norm_min, norm_max = 0, 1
norm = 127.5

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


for ir_img in img_list:
    img = Image.open(ir_img)
    img = img.convert('L')
    img_arr = np.array(img)
    input_data = img_arr.reshape(1, img_arr.shape[0], img_arr.shape[1], 1)
    if input_data.dtype != 'float32':
        input_data = (np.float32(input_data)-norm)/norm
        # input_data = np.float32(input_data)/max

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])  # threshold
    output_data1 = interpreter.get_tensor(output_details[1]['index'])  # location
    output_data2 = interpreter.get_tensor(output_details[2]['index'])  # total detected
    output_data3 = interpreter.get_tensor(output_details[3]['index'])  # score

    output_data4 = output_data1*79
    output_data4 = np.around(output_data4, decimals=0).astype(int).tolist()

    # print(f"output_data: {output_data}")
    # print(f"output_data1: {output_data1}")
    # print(f"output_data2: {output_data2}")
    # print(f"output_data3: {output_data3}")
    # print(f"output_data4: {output_data4}")

    output = []
    for i in output_data4[0]:
        if 0 in i: break
        else: output.append(i)

    if SHOW == 1:
        window_name = "out"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        image = cv2.imread(ir_img, cv2.COLOR_BGR2RGB)
        for i, j in zip(output, output_data[0]):
            cv2.rectangle(image, (i[1], i[0]), (i[3], i[2]), r, 1)
            cv2.putText(image, f"{j:.2f}", (i[1]-2, i[0]-1), cv2.FONT_HERSHEY_SIMPLEX, 0.2, g, 1)

        cv2.imshow(window_name, image)
        key = cv2.waitKey()
        if key == 27:
            cv2.destroyAllWindows()

    if SAVE == 1:
        out_path = f"output/{ir_img}"
        cv2.imwrite(out_path, img_arr)

    output.clear()
