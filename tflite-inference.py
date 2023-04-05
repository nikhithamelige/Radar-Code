import numpy as np
import tensorflow as tf

type_of_quantization = "Default"
model_path = f"saved-tflite-model/range-doppler-{type_of_quantization}.tflite"

range_doppler_features = np.load("data/npz_files/range_doppler_cfar_data.npz", allow_pickle=True)
x_data, y_data = range_doppler_features['out_x'], range_doppler_features['out_y']

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_index = input_details["index"]

data = x_data[2]
print(data.shape)
in_tensor = np.float32(data.reshape(1, data.shape[0], data.shape[1], 1))

interpreter.set_tensor(input_index, in_tensor)
interpreter.invoke()
classes = interpreter.get_tensor(output_details['index'])

print(classes)