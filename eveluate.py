from datetime import datetime
from argparse import ArgumentParser

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf


DEBUG = 1

parser = ArgumentParser()
parser.add_argument("-s", "--show", default=0, type=int, help="show image")
args = parser.parse_args()

if DEBUG == 1:
    args.show = 1

input_path = "model/10.png"
model_path = "model/v1.1.tflite"
ir_save_path = "model/000.png"

min, max = 0, 255
norm_min, norm_max = 0, 1
norm = 127.5

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img = Image.open(input_path)
img_arr = np.array(img)
input_data = img_arr.reshape(1, img_arr.shape[0], img_arr.shape[1], 1)
if input_data.dtype != 'float32':
    input_data = (np.float32(input_data)-norm)/norm
    # input_data = np.float32(input_data)/max

interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

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


train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))

print(train_dataset.element_spec)
print(test_dataset.element_spec)

model_name = "pplcntr_"+datetime.now().strftime("%Y%m%d-%H%M%S")
model.save(filepath=model_name, overwrite=True, include_optimizer=True, save_format='tf', signatures=None, options=None, save_traces=True, )


model.evaluate(test_x, test_y)
