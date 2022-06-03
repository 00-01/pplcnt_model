import tensorflow as tf


converter = tf.lite.TFLiteConverter.from_saved_model("Model")
model_no_quant_tflite = converter.convert()
MODEL_NO_QUANT_TFLITE = 'g_model.tflite'
open(MODEL_NO_QUANT_TFLITE, "wb").write(model_no_quant_tflite)
os.system('xxd -i  '+MODEL_NO_QUANT_TFLITE+' > model_tflite.cc')

cc = open('model.cc', 'w')
cc.write('#include "model.h"\n')
with open('model_tflite.cc', 'r') as f:
    while True:
        s = f.readline().replace("_tflite", "").replace("unsigned", "const unsigned").replace('const unsigned int', 'const int')
        if s == '':
            break
        cc.write(s)
cc.close()

interpreter = tf.lite.Interpreter(model_content=model_no_quant_tflite)
interpreter.allocate_tensors()

output = interpreter.get_output_details()[0]  # Model has single output.
input = interpreter.get_input_details()[0]  # Model has single input.

# #0
# input_data = tf.constant([1., 1., 1., 1., 0., 1., 1., 0., 1., 1., 0., 1., 1., 1., 1.], shape=[1, 15])
# interpreter.set_tensor(input['index'], input_data)
# interpreter.invoke()
# print(interpreter.get_tensor(output['index']))
#
# #1
# input_data = tf.constant([0., 1., 0., 1., 1., 0., 0., 1., 0., 0., 1., 0., 1., 1., 1.], shape=[1, 15])
# interpreter.set_tensor(input['index'], input_data)
# interpreter.invoke()
# print(interpreter.get_tensor(output['index']))
#
# #2
# input_data = tf.constant([1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1.], shape=[1, 15])
# interpreter.set_tensor(input['index'], input_data)
# interpreter.invoke()
# print(interpreter.get_tensor(output['index']))


# print(interpreter.get_tensor(input['index']))
# interpreter.get_tensor(output['index']).shape
# print(interpreter.get_tensor(output['index']))
# interpreter.get_signature_list()
# interpreter.get_output_details()

k = 10
