# Sound classification using Recurrent Neural Networks
This repository is a RNN implementation using Tensorflow, to classify audio clips of different lengths.
The input of the neural networks is not the raw sound, but the [MFCC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) features (20 features for 512 time steps).

## Sources
* this [repository](https://github.com/aqibsaeed/Urban-Sound-Classification/blob/master/Urban%20Sound%20Classification%20using%20RNN.ipynb) and this [notebook] (https://musicinformationretrieval.com/mfcc.html) helped me to understand the mfcc features extraction.
* this [post](https://danijar.com/variable-sequence-lengths-in-tensorflow/) explains how to take into account the variable length of the sequences.
