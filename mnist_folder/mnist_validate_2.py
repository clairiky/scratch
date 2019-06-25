
import numpy as np    # we're going to use numpy to process input and output data
import onnxruntime    # to inference ONNX models, we use the ONNX Runtime
import onnx
from onnx import numpy_helper
import urllib.request
import time
import glob
import os


### One way of loading ###
#Load sample inputs and outputs
# from mnist import MNIST

# mndata = MNIST('samples')

# images, labels = mndata.load_testing()

# print (type(images[0]))
# print (type(images))
# print (images[0])
# print (type(labels))


### ANother way
import gzip
import numpy as np

#test inputs
f = gzip.open('t10k-images-idx3-ubyte.gz', 'rb')

image_size = 28
num_images = 1

buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)

m = np.zeros([1,1,28,28], dtype=np.float32)
m[0,0,:,:] = np.reshape(data, ( 28, 28))
print(m.shape)
print (m.dtype)
print (type(data))


myinputs = []


myinputs.append(m)
print (myinputs)

#print (data)
#print(np.reshape(data,(28, 28)))
# reference outputs

f = gzip.open('t10k-labels-idx1-ubyte.gz', 'rb')

ref_num_images = 2

ref_buf = f.read(image_size * image_size * ref_num_images)
ref_data = np.frombuffer(ref_buf, dtype=np.uint8).astype(np.float32)

rm = np.zeros([1,2,28,28], dtype=np.float32)
rm[0,:,:,:] = np.reshape(ref_data, ( 2, 28, 28))
print(rm.shape)
#print(rm)
#test = numpy_helper.to_array(rm)
#print (test.shape)

print (type(ref_data))
#print (ref_data)
#print(np.reshape(ref_data,(28, 28)))

# #Inference using ONNX Runtime

# Run the model on the backend
session = onnxruntime.InferenceSession('model.onnx', None)

input_name = session.get_inputs()[0].name  

print('Input Name:', input_name)
outputs = session.run([], {input_name: m})[0] 

# print (outputs)
# print(int(np.argmax(np.array(outputs).squeeze(), axis=0)))
# import struct
# with gzip.open('t10k-images-idx3-ubyte.gz', 'rb') as f:
#     zero, data_type, dims = struct.unpack('>HBB', f.read(4))
#     print (struct.calcsize('>HBB'))
#     print (zero)
#     print (data_type)
#     print (dims)
#     shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
#     print(type(np.fromstring(f.read(), dtype=np.uint8).reshape(shape)))


# test_data_dir = 'test_data_set'
# test_data_num = 1
# # Load inputs
# inputs = []
# for i in range(test_data_num):
#     input_file = os.path.join(test_data_dir + '_{}'.format(i), 'input_0.pb')
#     tensor = onnx.TensorProto()
#     with open(input_file, 'rb') as f:
#         #print (type(f))
#         tensor.ParseFromString(f.read())

#         print(type(numpy_helper.to_array(tensor)))
#         inputs.append(numpy_helper.to_array(tensor))

# print (inputs)

# print('Loaded {} inputs successfully.'.format(test_data_num))
        
# # Load reference outputs

# ref_outputs = []
# for i in range(test_data_num):
#     output_file = os.path.join(test_data_dir + '_{}'.format(i), 'output_0.pb')
#     tensor = onnx.TensorProto()
#     with open(output_file, 'rb') as f:
#         tensor.ParseFromString(f.read())    
#         ref_outputs.append(numpy_helper.to_array(tensor))
        
# print('Loaded {} reference outputs successfully.'.format(test_data_num))

# #Inference using ONNX Runtime

# # Run the model on the backend
# session = onnxruntime.InferenceSession('mnist\\model.onnx', None)



# # get the name of the first input of the model
# input_name = session.get_inputs()[0].name  
# print('Input Name:', input_name)

# start = time.time()

# outputs = [session.run([], {input_name: inputs[i]})[0] for i in range(test_data_num)]

# #end timer
# end = time.time()

# print (outputs)
# result = []
# for i in outputs:
#     result.append(int(np.argmax(np.array(i).squeeze(), axis=0)))

# result_dict = {"result": result,
#           "time_in_sec": end - start}

# print (result_dict)

# ref_result = []
# for i in ref_outputs:
#     ref_result.append(int(np.argmax(np.array(i).squeeze(), axis=0)))

# ref_outputs_dict = {"result": ref_result}
# print (ref_result)

# print('Predicted {} results.'.format(len(outputs)))

# #Compare the results with reference outputs up to 4 decimal places
# for ref_o, o in zip(ref_outputs, outputs):
#     np.testing.assert_almost_equal(ref_o, o, 2)
#     print(ref_o)
#     print ("---------------------------------")
#     print(o)
# print (test_data_num)
# print(len(outputs))

# print (outputs)
# print (ref_outputs)  
# print('ONNX Runtime outputs are similar to reference outputs!')