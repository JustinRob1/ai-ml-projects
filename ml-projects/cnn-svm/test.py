import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras import backend as K
import numpy as np
from PIL import Image

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Resize the images to 7x7
new_size = (7, 7)
x_train_resized = np.array([np.array(Image.fromarray(img).resize(new_size)) for img in x_train])
x_test_resized = np.array([np.array(Image.fromarray(img).resize(new_size)) for img in x_test])

# Reshape the input data based on the backend
if K.image_data_format() == 'channels_first':
    x_train_resized = x_train_resized.reshape(x_train_resized.shape[0], 1, new_size[0], new_size[1])
    x_test_resized = x_test_resized.reshape(x_test_resized.shape[0], 1, new_size[0], new_size[1])
    input_shape = (1, new_size[0], new_size[1])
else:
    x_train_resized = x_train_resized.reshape(x_train_resized.shape[0], new_size[0], new_size[1], 1)
    x_test_resized = x_test_resized.reshape(x_test_resized.shape[0], new_size[0], new_size[1], 1)
    input_shape = (new_size[0], new_size[1], 1)

# Normalize the pixel values to the range [0, 1]
x_train_resized = x_train_resized.astype('float32') / 255
x_test_resized = x_test_resized.astype('float32') / 255

# Convert the labels to categorical
num_classes = 10
y_train_categorical = to_categorical(y_train, num_classes)
y_test_categorical = to_categorical(y_test, num_classes)

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
batch_size = 128
epochs = 10
model.fit(x_train_resized, y_train_categorical, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test_resized, y_test_categorical))

# Evaluate the model
score = model.evaluate(x_test_resized, y_test_categorical, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])