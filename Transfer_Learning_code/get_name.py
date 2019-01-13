# 获取 ckpt 模型的节点名称
import os
from tensorflow.python import pywrap_tensorflow
import tensorflow as tf
# model_path = './tensorflow_inception_graph.pb'
model_path = './inception_v3.ckpt'
if model_path.split('.')[-1] == 'ckpt':
    reader = pywrap_tensorflow.NewCheckpointReader(model_path)
    var_to_sahpe_map = reader.get_variable_to_shape_map()
    for key in var_to_sahpe_map:
        print('tensor name:', key)
        # print('value:', reader.get_tensor(key))
else:
    # 获取pb模型节点的名称
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        graph_def = tf.get_default_graph()
        graph_def = graph_def.as_graph_def()

        for node in graph_def.node:
            print('tensor_name:', node.name)
