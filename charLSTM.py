from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import os
import time
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Dropout, Bidirectional
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback, ModelCheckpoint
from itertools import permutations
import random
import sys
import io
text = open('novel.txt', 'rb').read().decode(encoding='utf-8').lower()
vocab = sorted(set(text))
char_indexes = dict((c, i) for i, c in enumerate(vocab))
index_char = dict((i, c) for i, c in enumerate(vocab))
text_as_int = np.array([char_indexes[c] for c in text])
seq_len = 40
step = 3
sequences = []
labels = []
for i in range(0, len(text) - seq_len, step):
    sequences.append(text[i: i + seq_len])
    labels.append(text[i + seq_len])
x = np.zeros((len(sequences), seq_len, len(vocab)), dtype=np.bool)
y = np.zeros((len(sequences), len(vocab)), dtype=np.bool)
for i, sentence in enumerate(sequences):
    for t, char in enumerate(sentence):
        x[i, t, char_indexes[char]] = 1
    y[i, char_indexes[labels[i]]] = 1

model = Sequential()
model.add(Embedding(input_dim=len(vocab), output_dim=20))
model.add(LSTM(160,return_sequences=True))
model.add(LSTM(160))
model.add(Dense(len(vocab)))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.01)
model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
def generator(sentence_list, next_word_list, batch_size):
    index = 0
    while True:
        x = np.zeros((batch_size, seq_len), dtype=np.int32)
        y = np.zeros((batch_size), dtype=np.int32)
        for i in range(batch_size):
            for t, w in enumerate(sentence_list[index % len(sentence_list)]):
                x[i, t] = char_indexes[w]
            y[i] = char_indexes[next_word_list[index % len(sentence_list)]]
            index = index + 1
        yield x, y

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def on_epoch_end(epoch, logs):
    if epoch+1 == 1 or epoch+1 == 10 or epoch+1 == 20 or epoch+1 == 40:
        print()
        print('----- Generating text after Epoch: %d' % epoch)
        for diversity in [0.03, 0.2, 0.5, 0.6, 0.8, 1.0, 1.2]:
            print('\nWITH TEMPERATURE :', diversity)

            generated = ''
            #sentence='it is true that it may be quite possible'
            sentence='it is true that it may be quite possible'
            #sentence='it is true that it may be '
            generated += sentence
            print('AND SEED TEXT: ' + sentence+"\n")
            sys.stdout.write(generated)
            

            for i in range(1500):
                x_pred = np.zeros((1, seq_len))
                for t, char in enumerate(sentence):
                    x_pred[0, t] = char_indexes[char]

                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = index_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()
    else:
        print()
        print('----- Not generating text after Epoch: %d' % epoch)

generate_text = LambdaCallback(on_epoch_end=on_epoch_end)
filepath = "weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss',verbose=1,save_best_only=True,mode='min')

model.fit_generator(generator(sequences, labels, 64),steps_per_epoch=int(len(sequences)/64) + 1,epochs=20,callbacks=[generate_text, checkpoint])
