import numpy as np
import h5py
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.callbacks import ModelCheckpoint

if __name__ == '__main__':

    # Open and read data from the file    
    data = open('data/sonnets.txt', 'r').read()
    chars = sorted(list(set(data)))
    data_size, vocab_size = len(data), len(chars)
    print('There are %d total characters and %d unique characters in the data.' % (data_size, vocab_size))

    # Create dictionaries mapping individual characters to indices and vice versa
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }
    print(ix_to_char)
    print(char_to_ix)

    # Cut the text in semi-redundant sequences of maxlen characters
    maxLen = 50
    step = 1
    sentences = []
    next_chars = []

    for i in range(0, len(data)-maxLen, step):
        sentences.append(data[i:i+maxLen])
        next_chars.append(data[i+maxLen])
    
    print('Training data length:%d') % len(sentences)

    # Vectoring data
    print('Vectorizing data...')
    X = np.zeros((len(sentences), maxLen, len(chars)), dtype=np.bool)
    Y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_to_ix[char]] = 1
        Y[i, char_to_ix[next_chars[i]]] = 1

    # Build the LSTM model:
    print('Build the LSTM model')
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(maxLen, len(chars))))
    model.add(LSTM(128))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Dump the model architecture to a file
    model_architecture = model.to_yaml()
    with open('model_architecture.yaml', 'a') as model_file:
        model_file.write(model_architecture)

    file_path='weights.{epoch:02d}-{acc:.2f}.hdf5'
    checkpoint = ModelCheckpoint(file_path, monitor="acc", verbose=1, save_weights_only=True, save_best_only=True, mode="max")
    callbacks = [checkpoint]
    # Fit the model
    model.fit(X, Y, epochs = 100, batch_size = 128, callbacks=callbacks)  
