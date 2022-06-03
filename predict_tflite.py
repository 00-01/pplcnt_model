import argparse
import time

import numpy as np
from PIL import Image
import tensorflow as tf
import tflite_runtime.interpreter as tflite


def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', default='./model/1651097056663.png', help='image to be classified')
    parser.add_argument('-m', '--model_file', default='./model/v1.1.tflite', help='.tflite model to be executed')
    parser.add_argument('-l', '--label_file', default='./model/class.txt', help='name of file containing labels')
    parser.add_argument('--input_mean', default=127.5, type=float, help='input_mean')
    parser.add_argument('--input_std', default=127.5, type=float, help='input standard deviation')
    parser.add_argument('--num_threads', default=None, type=int, help='number of threads')
    parser.add_argument('-e', '--ext_delegate', help='external_delegate_library path')
    parser.add_argument('-o', '--ext_delegate_options', help='external delegate options, format: "option1: value1; option2: value2"')
    args = parser.parse_args()

    ext_delegate = None
    ext_delegate_options = {}

    # parse extenal delegate options
    if args.ext_delegate_options is not None:
        options = args.ext_delegate_options.split(';')
        for o in options:
            kv = o.split(':')
            if (len(kv) == 2): ext_delegate_options[kv[0].strip()] = kv[1].strip()
            else: raise RuntimeError('Error parsing delegate option: '+o)

    # load external delegate
    if args.ext_delegate is not None:
        print(f'Loading external delegate from {args.ext_delegate} with args: {ext_delegate_options}')
        ext_delegate = [tflite.load_delegate(args.ext_delegate, ext_delegate_options)]

    interpreter = tf.lite.Interpreter(model_path=args.model_file, experimental_delegates=ext_delegate, num_threads=args.num_threads)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    img = Image.open(args.image).resize((width, height))

    # add N dim
    input_data = np.expand_dims(img, axis=0)
    # img = np.expand_dims(img_arr, axis=-1)
    # img = img_arr.reshape(img_arr.shape[0], img_arr.shape[1], 1)
    input_data = input_data.reshape(input_data.shape[0], input_data.shape[1], input_data.shape[2], 1)

    if floating_model: input_data = (np.float32(input_data)-args.input_mean)/args.input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)

    start_time = time.time()
    interpreter.invoke()
    stop_time = time.time()-start_time

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    top_k = results.argsort()[-2:][::-1]
    labels = load_labels(args.label_file)
    for i in top_k:
        if floating_model:
            print(f'{float(results[i]):08.6f}: {labels[i]}')
        else:
            print(f'{float(results[i]/255.0):08.6f}: {labels[i]}')

    print(f'time: {stop_time*1000:.2f}ms')


# find output from this
# output_data =  [{'name': 'StatefulPartitionedCall:1', 'index': 134, 'shape': array([ 1, 10], dtype=int32), 'shape_signature': array([ 1, 10], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}},
#                 {'name': 'StatefulPartitionedCall:3', 'index': 132, 'shape': array([ 1, 10,  4], dtype=int32), 'shape_signature': array([ 1, 10,  4], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}},
#                 {'name': 'StatefulPartitionedCall:0', 'index': 135, 'shape': array([1], dtype=int32), 'shape_signature': array([1], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}},
#                 {'name': 'StatefulPartitionedCall:2', 'index': 133, 'shape': array([ 1, 10], dtype=int32), 'shape_signature': array([ 1, 10], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}]

