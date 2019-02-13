"""
    - train a CNN using the encoded faetures from the DAE ( uncomment 'train_CNN' function )
    - classify a query image
"""


import tensorflow as tf
from keras.layers import Input, MaxPooling2D, Dense, Conv2D, Flatten
from keras.models import Model, load_model
from keras.datasets import mnist
import numpy as np
from keras.utils import np_utils
import cv2


def train_CNN(x, y, x_val, y_val, input_shape):

    features = Input(shape = input_shape)
    X = Conv2D(filters = 8, kernel_size = (5, 5), padding = 'same')(features)
    X = Conv2D(filters = 8, kernel_size = (5, 5), padding = 'same', activation = 'relu')(X)
    X = MaxPooling2D(pool_size = (2, 2))(X)
    X = Conv2D(filters = 16, kernel_size = (3, 3), padding = 'same')(X)
    X = Conv2D(filters = 16, kernel_size = (3, 3), padding = 'same', activation = 'relu')(X)
    X = MaxPooling2D(pool_size = (2, 2))(X)
    X = Dense(units = 64, activation = 'relu')(X)
    X = Flatten()(X)
    pred = Dense(units = 10, activation = 'sigmoid', name = 'CNN_pred')(X)

    CNN = Model(inputs = features, outputs = pred)

    CNN.compile(optimizer='adadelta', loss = 'binary_crossentropy', metrics=['acc'])

    CNN.fit(x, y, epochs = 5, batch_size = 128, shuffle = True, validation_data = (x_val, y_val))

    CNN.save('CNN.h5')

def classify(query_image, encoder):
    resized_query_image = cv2.resize(query_image, (280, 280))
    cv2.imshow('query_image', resized_query_image)
    cv2.waitKey(0)
    image_code = encoder.predict(query_image.reshape(1, query_image.shape[0], query_image.shape[1], 1))
    CNN = load_model('CNN.h5')
    print('predicted class: ', np.argmax(CNN.predict(image_code)))



# Load mnist dataset
print('loading mnist dataset...')

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Load previously trained autoencoder
print('Loading model...')
autoencoder = load_model('autoencoder.h5')

# Build the encoder model used to find the images Representation/Features
encoder = Model(inputs = autoencoder.input, outputs = autoencoder.get_layer('encoded').output)

# Compute Representation/Features for the training images
learned_codes = encoder.predict(x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))

# Compute Representation/Features for the test dataset
test_codes = encoder.predict(x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

num_classes = 10
y_train_ohe = np_utils.to_categorical(y_train, num_classes)
y_test_ohe = np_utils.to_categorical(y_test, num_classes)

# train the CNN model using the learned codes
# train_CNN(x = learned_codes, y = y_train_ohe, x_val = test_codes, y_val = y_test_ohe, input_shape = (7, 7, 8) )

# classify a certain image
classify(query_image = x_test[1], encoder = encoder)
