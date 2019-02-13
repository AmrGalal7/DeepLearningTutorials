import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.datasets import mnist
import numpy as np


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

def train_model():
    input_img = Input(shape = (28, 28, 1))
    x = Conv2D(filters = 16, kernel_size = (5, 5), padding = 'same')(input_img)
    x = Conv2D(16, (5, 5), padding = 'same', activation = 'relu')(x)
    x = MaxPooling2D(pool_size = (2, 2))(x)
    x = Conv2D(8, (3, 3), padding = 'same')(x)
    x = Conv2D(8, (3, 3), activation = 'relu', padding = 'same')(x)
    encoded = MaxPooling2D(pool_size = (2, 2), name = 'encoded')(x)
    # encoded shape: [m x 7 x 7 x 8]

    x = Conv2D(8, (3, 3), padding = 'same')(encoded)
    x = Conv2D(8, (3, 3), padding = 'same', activation = 'relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (5, 5), padding = 'same')(x)
    x = Conv2D(16, (5, 5), padding = 'same', activation = 'relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (5, 5), activation = 'sigmoid', padding = 'same')(x)
    # encoded shape: [m x 28 x 28 x 1]


    autoencoder = Model(inputs = input_img, outputs = decoded)
    autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')

    autoencoder.fit(x_train_noisy, x_train, epochs = 3, batch_size = 128, \
                    shuffle = True, validation_data = (x_test_noisy, x_test), \
                    callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])

    autoencoder.save('autoencoder.h5')

train_model()
