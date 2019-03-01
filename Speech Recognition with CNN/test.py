from keras.models import load_model
from dataset import DatasetGenerator
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
import matplotlib.pyplot as plt

DIR = '/home/amrgalal7/Documents/speech_commands_v0.01' # unzipped train and test data

INPUT_SHAPE = (177,98,1)
BATCH = 32
EPOCHS = 15

LABELS = 'yes no up'.split()
NUM_CLASSES = len(LABELS)


def plot_model_history(path_to_history):
    '''
        visualises model history
    '''
    with open(path_to_history, 'rb') as f:
        history = pickle.load(f)
    # model history for accuracy
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # model history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


dsGen = DatasetGenerator(label_set=LABELS)
# load DataFrame with paths/labels
df = dsGen.load_data(DIR)
dsGen.apply_train_test_val_split(0.7, 0.7)

# load the model
model = load_model('cnn_4.h5')

# load and visualise model history
plot_model_history('history_4')

y_pred_proba = model.predict_generator(dsGen.generator(BATCH, mode='test'),
                                        int(np.ceil(len(dsGen.df_test)/BATCH)), verbose=1)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = dsGen.df_test['label_id'].values
acc_score = accuracy_score(y_true, y_pred)
print('model accuracy: ', acc_score)
