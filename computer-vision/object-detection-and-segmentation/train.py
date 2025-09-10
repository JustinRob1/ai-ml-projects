import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Load the dataset
images = np.load('train_X.npy')  # replace with your file path
masks = np.load('train_seg.npy')  # replace with your file path

# Preprocess the images and masks
images = images.reshape((-1, 64, 64, 3)).astype(np.float32) / 255
masks = masks.reshape((-1, 64, 64, 1)).astype(np.int32)

images = images[:10000]
masks = masks[:10000]

# Split the dataset into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

# Define the UNet model
def create_model():
    inputs = Input((64, 64, 3))
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)

    up4 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(up4)

    up5 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=3)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(up5)

    output = Conv2D(11, 1, activation='softmax')(conv5)

    model = Model(inputs=[inputs], outputs=[output])
    return model

# Create and compile the model
model = create_model()
model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32)

# Save the model
model.save('unet_model.h5')