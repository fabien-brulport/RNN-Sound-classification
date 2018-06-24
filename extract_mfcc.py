import librosa
import glob
import os
import numpy as np
import pandas as pd
import pickle


def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size // 2)


def extract_features(parent_dir, label_file=None, file_ext="*.wav", bands=20,
                     frames=41, kind='train'):
    if kind == 'train':
        window_size = 512 * (frames - 1)
        df = pd.read_csv(label_file)
        df = df.set_index('fname')
        mfccs = []
        labels = []
        i = 0
        for fn in glob.glob(os.path.join(parent_dir, file_ext)):
            sound_clip, s = librosa.load(fn)
            label = df.loc[fn[-12:]].label
            for (start, end) in windows(sound_clip, window_size):
                if (len(sound_clip[start:end]) == window_size):
                    signal = sound_clip[start:end]
                    mfcc = librosa.feature.mfcc(y=signal, sr=s,
                                                n_mfcc=bands).T.flatten()[:,
                           np.newaxis].T
                    mfccs.append(mfcc)
                    labels.append(label)
            i += 1
            if i % 10 == 0:
                print(i)
        features = np.asarray(mfccs).reshape(len(mfccs), frames, bands)

        with open('./input/mfcc_features.p', 'wb') as fp:
            pickle.dump(np.array(features), fp)
        with open('./input/mfcc_labels.p', 'wb') as fp:
            pickle.dump(np.array(labels), fp)

    elif kind == 'test':
        window_size = 512 * (frames - 1)
        mfccs = []
        fname = []
        for i, fn in enumerate(glob.glob(os.path.join(parent_dir, file_ext))):
            sound_clip, s = librosa.load(fn)
            sample_mfcc = []
            for (start, end) in windows(sound_clip, window_size):
                if len(sound_clip[start:end]) == window_size:
                    signal = sound_clip[start:end]
                    mfcc = librosa.feature.mfcc(y=signal, sr=s,
                                                n_mfcc=bands).T.flatten()[:,
                           np.newaxis].T
                    sample_mfcc.append(
                        np.asarray(mfcc).reshape((frames, bands)))
                if start == 0 and end > len(sound_clip)-1:
                    signal = np.zeros(window_size)
                    signal[0:len(sound_clip)] = sound_clip
                    mfcc = librosa.feature.mfcc(y=signal, sr=s,
                                                n_mfcc=bands).T.flatten()[:,
                           np.newaxis].T
                    sample_mfcc.append(
                        np.asarray(mfcc).reshape((frames, bands)))
            fname.append(fn)
            mfccs.append(sample_mfcc)
            if i % 10 == 0:
                print('{} == {}'.format(i + 1, len(mfccs)))
        with open('./input/mfcc_test.p', 'wb') as fp:
            pickle.dump(np.array(mfccs), fp)
        with open('./input/mfcc_test_name.p', 'wb') as fp:
            pickle.dump(fname, fp)


if __name__ == '__main__':
    dir = './input/audio_test'
    label_f = './input/train.csv'
    extract_features(dir, label_f, kind='test')



