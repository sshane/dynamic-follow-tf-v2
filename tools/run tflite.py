import numpy as np
import tensorflow as tf
import os

os.chdir("C:/Git/dynamic-follow-tf/models")
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']

#input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
input_data = np.asarray([[32, 10, 81]], dtype=np.float32)

print(input_data)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
