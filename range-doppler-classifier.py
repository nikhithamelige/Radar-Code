import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

range_doppler_features = np.load("data/range_doppler_data.npz", allow_pickle=True)

x_data, y_data = range_doppler_features['out_x'], range_doppler_features['out_y']

classes_values = ["empty_room", "occupied_room"]
classes = len(classes_values)

y_data = tf.keras.utils.to_categorical(y_data - 1, classes)

train_ratio = 0.90

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=1 - train_ratio)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=(16, 128, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (2, 2), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (2, 2), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(classes, activation='softmax')
])

# model.summary()
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['acc'])

# this controls the batch size
BATCH_SIZE = 60
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False)
validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=False)

history = model.fit(train_dataset, epochs=100, validation_data=validation_dataset)

# model.save(f"saved-model/range-doppler-model")

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

print(f"Training Accuracy: {round(np.average(acc), 3)}")
print(f"Validation Accuracy: {round(np.average(val_acc), 3)}")

epochs = range(1, len(acc) + 1)
fig, axs = plt.subplots(2, 1)

# plot loss
axs[0].plot(epochs, loss, '-', label='Training loss')
axs[0].plot(epochs, val_loss, 'b', label='Validation loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].grid(True)
axs[0].legend(loc='best')
# plot accuracy
axs[1].plot(epochs, acc, '-', label='Training acc')
axs[1].plot(epochs, val_acc, 'b', label='Validation acc')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].grid(True)
axs[1].legend(loc='best')
plt.show()
