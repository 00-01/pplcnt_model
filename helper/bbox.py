from argparse import ArgumentParser

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf


DEBUG = 1

parser = ArgumentParser()
# parser.add_argument("-l", "--loop", default=0, type=int, help="run loop")
# parser.add_argument("-s", "--sleep", default=0, type=int, help="loop sleep")
# parser.add_argument("-o", "--offset", default=0, type=int, help="offset")
# parser.add_argument("-b", "--box", default=0, type=int, help="draw box")
# parser.add_argument("-t", "--transform", default=0, type=int, help="transform")
# parser.add_argument("-min", "--min", default=0, type=int, help="min")
parser.add_argument("-s", "--show", default=0, type=int, help="show image")
args = parser.parse_args()

if DEBUG == 1:
    args.show = 1

input_path = "../model/10.png"
model_path = "../model/v1.1.tflite"
ir_save_path = "../model/000.png"

min, max = 0, 255
norm_min, norm_max = 0, 1
norm = 127.5

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img = Image.open(input_path)  # .resize((width, height))
img_arr = np.array(img)
input_data = img_arr.reshape(1, img_arr.shape[0], img_arr.shape[1], 1)
if input_data.dtype != 'float32':
    input_data = (np.float32(input_data)-norm)/norm
    # input_data = np.float32(input_data)/max

interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# function `get_tensor()` returns copy of tensor data. Use `tensor()`  to get pointer to tensor
output_data = interpreter.get_tensor(output_details[0]['index'])
output_data1 = interpreter.get_tensor(output_details[1]['index'])
output_data2 = interpreter.get_tensor(output_details[2]['index'])
output_data3 = interpreter.get_tensor(output_details[3]['index'])

output_data4 = output_data1*79
output_data4 = np.around(output_data4, decimals=0).astype(int).tolist()
output = []
for i in output_data4:
    for j in i:
        if 0 in j: break
        else: output.append(j)

print(f"output_data: {output_data}")
print(f"output_data1: {output_data1}")
print(f"output_data2: {output_data2}")
print(f"output_data3: {output_data3}")
print(f"output_data4: {output_data4}")

if args.show == 1:
    window_name = "out"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    for i in output:
        image = cv2.rectangle(image, (i[1],i[0]), (i[3],i[2]), (255,0,0), 1)
    cv2.imshow(window_name, image)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()
