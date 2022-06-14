import tensorflow as tf


model = tf.saved_model.load('./model/saved_model')
# model = tf.keras.models.load_model('./model/saved_model/')

# Check its architecture
# model.summary()
# Predict model
# model.predict("./model/3.png")
a = 10

# from tensorflow.python.platform import gfile
#
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
#
# GRAPH_PB_PATH = './model/saved_model.pb'
#
# print("load graph")
# with gfile.FastGFile(GRAPH_PB_PATH, 'rb') as f:
#     graph_def = tf.GraphDef()
# graph_def.ParseFromString(f.read())
# sess.graph.as_default()
# tf.import_graph_def(graph_def, name='')
# graph_nodes = [n for n in graph_def.node]
# names = []
# for t in graph_nodes:
#     names.append(t.name)
# print(names)