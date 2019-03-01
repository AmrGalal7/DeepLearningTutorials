import os
import numpy as np
import pandas as pd
import random
from glob import glob
from scipy.io import wavfile
from scipy.signal import stft
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

class DatasetGenerator():
    def __init__(self, label_set, sample_rate = 16000):
        self.label_set = label_set
        self.sample_rate = sample_rate

    def text_to_labels(self, text):
        '''
            returns the index/label of a class name
        '''
        return self.label_set.index(text)

    def label_to_text(self, labels):
        '''
            returns the class(es) name(s) of the given indices/labels
        '''
        return self.label_set[labels]

    def load_data(self, DIR):
        # get the full paths of all .wav files in 'DIR'
        wav_files = glob(os.path.join(DIR, '*/*wav'))
        # extract the 'class_name/file_name.wav' path of each file in 'wav_files'
        wav_files = [ x.split(DIR + '/')[1] for x in wav_files ]

        data = []
        for wav in wav_files:
            # separate 'class_name' from 'file_name' for each 'class_name/file_name.wav'
            label, user = wav.split('/')
            if label in self.label_set:
                label_id = self.text_to_labels(label)
                file_extension = os.path.join(DIR + '/' + wav)
                sample = (label, label_id, user, file_extension)
                data.append(sample)

        # Data Frames with the samples' labels and paths
        df = pd.DataFrame(data, columns = ['label', 'label_id', 'user_id', 'wav_file'])
        self.df = df
        return self.df

    # def apply_train_test_split(self, test_size, random_state):
    #     self.df_train, self.df_test = train_test_split(self.df, test_size = test_size, random_state = random_state)
    #
    # def apply_train_val_split(self, val_size, random_state):
    #     self.df_train, self.df_val = train_test_split(self.df_train, test_size = val_size, random_state = random_state)

    def apply_train_test_val_split(self, train_size, val_size):
        '''
            split the data while avoiding the existence of the same user in training and testing at the same time

            Parameters:
            -----------
                    train_size: percentage of the data to be used for training
                    val_size: percentage of the test data to be used for validation
                        --> (1 - train_size) * (1 - val_size) * data is the testing data
        '''
        # get unique users
        users = self.df.user_id.drop_duplicates()
        # shuffle
        np.random.seed(42)
        shuffled_indices = np.random.permutation(users.shape[0])
        users = users.iloc[shuffled_indices]

        # split the users
        # surely, 0.7 of users isn't necessarily 0.7 of audio files!
        i = int(train_size * users.shape[0])
        train_users = users.iloc[:i]
        self.df_train = self.df[self.df.user_id.isin(train_users)]

        test_users = users.iloc[i:]
        j = int(0.7 * test_users.shape[0])
        valid_users = test_users.iloc[:j]
        test_users = test_users.iloc[j:]
        self.df_val = self.df[self.df.user_id.isin(valid_users)]
        self.df_test = self.df[self.df.user_id.isin(test_users)]

    def read_wav_file(self, f):
        _, wav = wavfile.read(f)
        # normalise
        wav = wav.astype(np.float32) / np.iinfo(wav.dtype).max
        return wav

    def process_wav_file(self, f, threshold_freq = 5500, eps = 1e-10):
        wav = self.read_wav_file(f)
        # use 1 second --> samples in 1 second =  sample rate * 1 second
        length = self.sample_rate
        # if file length > 'length', randomly extract 'length' from the file
        if len(wav) > length:
            start = np.random.randint(0, len(wav) - length)
            wav = wav[start: start + length]
        # if file length < 'length', randomly add silence
        elif len(wav) < length:
            remaining_length = length - len(wav)
            silence = np.random.randint(-1, 1, self.sample_rate)
            silence = silence.astype(np.float32) / np.iinfo(silence.dtype).max
            i = np.random.randint(0, remaining_length)
            silence_before_word = silence[0: i]
            silence_after_word = silence[i: remaining_length]
            wav = np.concatenate([silence_before_word, wav, silence_after_word])
        # create spectrogram
        freqs, times, spec = stft(wav, length, nperseg = 400, noverlap = 240, nfft = 512, padded = False, boundary = None)
        # cut high frequencies
        if threshold_freq is not None:
            spec = spec[freqs <= threshold_freq,:]
            freqs = freqs[freqs <= threshold_freq]
        # Log spectrogram
        amp = np.log(np.abs(spec) + eps)

        return np.expand_dims(amp, axis = 2)

    def generator(self, batch_size, mode):
        while True:
            # select the proper dataframe
            if(mode == 'train'):
                df = self.df_train
                ids = random.sample( range(df.shape[0]), df.shape[0] )
            elif(mode == 'test'):
                df = self.df_test
                ids = list(range(df.shape[0]))
            elif(mode == 'val'):
                df = self.df_val
                ids = list(range(df.shape[0]))
            else:
                raise ValueError('The mode should be either train, val or test.')

            # create batches (for training data the batches are randomly permuted)
            for start in range(0, len(ids), batch_size):
                X_batch = []
                if mode != 'test':
                    y_batch = []
                end = min(start + batch_size, len(ids))
                i_batch = ids[start:end]
                for i in i_batch:
                    X_batch.append( self.process_wav_file(df.wav_file.values[i]) )
                    if mode != 'test':
                        y_batch.append(df.label_id.values[i])
                X_batch = np.array(X_batch)

                if mode != 'test':
                    y_batch = to_categorical(y_batch, num_classes = len(self.label_set))
                    yield (X_batch, y_batch)
                else:
                    yield X_batch
