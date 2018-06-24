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


def extract_features(parent_dir, label_file, file_ext="*.wav", bands=20,
                     frames=41):
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
        if i % 10 ==0:
            print(i)
    features = np.asarray(mfccs).reshape(len(mfccs), frames, bands)
    return np.array(features), np.array(labels)


if __name__ == '__main__':
    dir = './input/audio_train'
    label_f = './input/train.csv'
    tr_features, tr_labels = extract_features(dir, label_f)
    with open('./input/mfcc.p', 'wb') as fp:
        pickle.dump((tr_features, tr_labels), fp)
