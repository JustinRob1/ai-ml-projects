# Source: https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-fashion-mnist-clothing-classification/
# Soruce: https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/

import numpy as np
from sklearn.model_selection import KFold
from keras.datasets import fashion_mnist, mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers.legacy import SGD
from PIL import Image

# load train and test dataset
def load_dataset_fashion():
    # load dataset
    (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
    # reshape dataset to have a single channel
    train_x = train_x.reshape((train_x.shape[0], 28, 28, 1))
    test_x = test_x.reshape((test_x.shape[0], 28, 28, 1))
    # one hot encode target values
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    return train_x, train_y, test_x, test_y

# Load and resize train and test dataset for low-res MNIST 7x7
def load_dataset_low_res():
    # load dataset
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    # resize dataset to 7x7
    train_x = np.array([np.array(Image.fromarray(img).resize((7,7))) for img in train_x])
    test_x = np.array([np.array(Image.fromarray(img).resize((7,7))) for img in test_x])
    train_x = train_x.reshape(train_x.shape[0], 7, 7, 1)
    test_x = test_x.reshape(test_x.shape[0], 7, 7, 1)
    # one hot encode target values
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    return train_x, train_y, test_x, test_y

# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm

# define cnn model
def define_model(dataset):
    model = Sequential()
    if dataset == "low_res_mnist":
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform', input_shape=(7, 7, 1)))
    else:
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# evaluate a model using k-fold cross-validation
def evaluate_model(data_x, data_y, n_folds=2, n_epochs=3, dataset="fashion_mnist"):
    scores, histories = list(), list()
    # prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    # enumerate splits
    for train_ix, test_ix in kfold.split(data_x):
        print("Running fold")
        # define model
        model = define_model(dataset)
        # select rows for train and test
        train_x, train_y, test_x, test_y = data_x[train_ix], data_y[train_ix], data_x[test_ix], data_y[test_ix]
        # fit model
        history = model.fit(train_x, train_y, epochs=n_epochs, batch_size=32, validation_data=(test_x, test_y), verbose=0)
        # evaluate model
        _, acc = model.evaluate(test_x, test_y, verbose=0)
        print('> %.3f' % (acc * 100.0))
        # append scores
        scores.append(acc)
        histories.append(history)
    return scores, histories

# run the test harness for evaluating a model
def run_test(dataset):
    print("Training on dataset: ", dataset)
    # load dataset
    if dataset == "low_res_mnist":
        train_x, train_y, test_x, test_y = load_dataset_low_res()
    else:
        train_x, train_y, test_x, test_y = load_dataset_fashion()
    # prepare pixel data
    train_x, test_x = prep_pixels(train_x, test_x)
    # evaluate model
    scores, histories = evaluate_model(train_x, train_y, n_folds=5, n_epochs=10, dataset=dataset)
    
    # summarize estimated performance
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (np.mean(scores)*100, np.std(scores)*100, len(scores)))
    
    return histories
