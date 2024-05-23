import tensorflow as tf
import numpy as np
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print("Is GPU available:", tf.test.is_gpu_available())
print("Built with CUDA:", tf.test.is_built_with_cuda())

physical_devices = tf.config.list_physical_devices('GPU')
print("GPUs:", physical_devices)
# Create a simple dataset
data = np.random.random((100000, 128))
labels = np.random.random((100000, 10))

# Define a simple model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1048, activation='relu'),
    tf.keras.layers.Dense(2096, activation='relu'),
    tf.keras.layers.Dense(1048, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(data, labels, epochs=100)
