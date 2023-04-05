import tensorflow as tf
import numpy as np


type_of_quantization = "default"
saved_model_dir = "saved-model/range-doppler-model"
BATCH_SIZE = 65

range_doppler_features = np.load("data/npz_files/range_doppler_cfar_data.npz", allow_pickle=True)

x_data, y_data = range_doppler_features['out_x'], range_doppler_features['out_y']


def representative_dataset():
    range_doppler = tf.data.Dataset.from_tensor_slices(x_data).batch(1)
    for i in range_doppler.take(BATCH_SIZE):
        yield [i]


# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)  # path to the SavedModel directory
if type_of_quantization == "default":
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the model.
    with open('saved-tflite-model/range-doppler-default.tflite', 'wb') as f:
        f.write(tflite_model)

elif type_of_quantization == "int8":
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8  # or tf.uint8
    converter.inference_output_type = tf.uint8  # or tf.uint8
    tflite_model = converter.convert()

    # Save the model.
    with open('saved-tflite-model/range-doppler-int8.tflite', 'wb') as f:
        f.write(tflite_model)

elif type_of_quantization == "float16":
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    # Save the model.
    with open('saved-tflite-model/range-doppler-float16.tflite', 'wb') as f:
        f.write(tflite_model)
