import numpy as np
from keras.callbacks import EarlyStopping
from dataset import DatasetGenerator
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.callbacks import TensorBoard
import pickle

DIR = '/home/amrgalal7/Documents/speech_commands_v0.01' # unzipped train and test data

INPUT_SHAPE = (177,98,1)
BATCH = 32
EPOCHS = 15

LABELS = 'yes no up'.split()
NUM_CLASSES = len(LABELS)

dsGen = DatasetGenerator(label_set=LABELS)
# Load DataFrame with paths/labels
df = dsGen.load_data(DIR)
# dsGen.apply_train_val_split(val_size=0.2, random_state=2018)
dsGen.apply_train_test_val_split(0.7, 0.7)

def deep_cnn(features_shape, num_classes, act='relu'):

    x = Input(name='inputs', shape=features_shape, dtype='float32')
    o = x

    # Block 1
    o = Conv2D(8, (11, 11), activation=act, padding='same', strides=1, name='block1_conv', input_shape=features_shape)(o)
    o = MaxPooling2D((3, 3), strides=(2,2), padding='same', name='block1_pool')(o)
    o = BatchNormalization(name='block1_norm')(o)

    # Block 2
    o = Conv2D(16, (7, 7), activation=act, padding='same', strides=1, name='block2_conv')(o)
    o = MaxPooling2D((3, 3), strides=(2,2), padding='same', name='block2_pool')(o)
    o = BatchNormalization(name='block2_norm')(o)

    # Block 3
    o = Conv2D(32, (5, 5), activation=act, padding='same', strides=1, name='block3_conv')(o)
    o = MaxPooling2D((3, 3), strides=(2,2), padding='same', name='block3_pool')(o)
    o = BatchNormalization(name='block3_norm')(o)

    # Block 4
    o = Conv2D(32, (3, 3), activation=act, padding='same', strides=1, name='block4_conv')(o)
    o = MaxPooling2D((3, 3), strides=(2,2), padding='same', name='block4_pool')(o)
    o = BatchNormalization(name='block4_norm')(o)

    # Flatten
    o = Flatten(name='flatten')(o)

    # Dense layer
    o = Dense(64, activation=act, name='dense')(o)
    o = BatchNormalization(name='dense_norm')(o)
    o = Dropout(0.2, name='dropout')(o)

    # Predictions
    o = Dense(num_classes, activation='softmax', name='pred')(o)

    # Print network summary
    Model(inputs=x, outputs=o).summary()

    return Model(inputs=x, outputs=o)


model = deep_cnn(INPUT_SHAPE, NUM_CLASSES)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])

callbacks = [EarlyStopping(monitor='val_acc', patience = 3, verbose=1, mode='max'), TensorBoard(log_dir='/tmp/tb_4', histogram_freq=0, write_graph=False)]

history = model.fit_generator(generator=dsGen.generator(BATCH, mode='train'),
                              steps_per_epoch=int(np.ceil(len(dsGen.df_train)/BATCH)),
                              epochs=EPOCHS,
                              verbose=1,
                              callbacks=callbacks,
                              validation_data=dsGen.generator(BATCH, mode='val'),
                              validation_steps=int(np.ceil(len(dsGen.df_val)/BATCH)))

model.save('cnn_4.h5')
with open('history_4', 'wb') as f:
    pickle.dump(history.history, f)
