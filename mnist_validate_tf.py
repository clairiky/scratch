
import tensorflow as tf
import os
import sys
from tensorflow.python.platform import gfile
from PIL import Image
import numpy as np

model = sys.argv[1]
input_dir = sys.argv[2]
output_dir = sys.argv[3]

sess = tf.Session()
graph_def = tf.GraphDef()

f = gfile.FastGFile(model, 'rb')
# Parses a serialized binary message into the current message.
graph_def.ParseFromString(f.read())
f.close()

sess.graph.as_default()
# Import a serialized TensorFlow `GraphDef` protocol buffer
# and place into the current default `Graph`.
tf.import_graph_def(graph_def)

#Get the node names
#print ([n.name for n in sess.graph.as_graph_def().node])

softmax_tensor = sess.graph.get_tensor_by_name('import/dense_2/Softmax:0')
ref_output = {}
with open(f'{output_dir}\\result.txt') as f:
    lineList = f.readlines()

for a in lineList:
    ref_output[a.split(',')[0]] = a.split(',')[1].strip()

images = {}
for a in range(100):
    images[f'img_{a}.png'] = Image.open(f'{input_dir}\\img_{a}.png')

top1_accuracy = 0
for img in images:
    prediction = sess.run(softmax_tensor, {'import/conv2d_1_input:0': np.array(images[img]).astype('float32').reshape(1,28,28,1)})
    print (prediction.argmax())
    if int(ref_output[img]) == int(prediction.argmax()):
        top1_accuracy +=1

print (top1_accuracy/len(images) * 100)