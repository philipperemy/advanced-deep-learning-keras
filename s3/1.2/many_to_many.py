import random
import sys

import numpy as np
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.core import RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.utils.data_utils import get_file

path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path, 'rb').read().decode('utf8').lower()
print('corpus length:', len(text))  # corpus length: 600893

chars = sorted(list(set(text)))
print('total chars:', len(chars))  # total chars: 57
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen:i + maxlen + 3])
print('nb sequences:', len(sentences))  # nb sequences: 200285

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), 3, len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1

for i, sentence in enumerate(next_chars):
    for t, char in enumerate(sentence):
        y[i, t, char_indices[char]] = 1

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(RepeatVector(3))
model.add(TimeDistributed(Dense(len(chars), activation='softmax')))

#############################################
# Process sequence and output a vector of length 128 (after LSTM)
# Duplicate this vector of length 128 into 3 vectors
# Apply a Dense layer on each one (TimeDistributed).
# For each output of the 3 vectors, apply a softmax to find the best
# next character. This way we can predict 3 characters in advance.
#############################################

model.compile(loss='categorical_crossentropy', optimizer='adam')


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# train the model, output generated text after each iteration
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y,
              batch_size=128,
              epochs=1)

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            pred_chars = model.predict(x, verbose=0)[0]
            next_indexes = [sample(pred, diversity) for pred in pred_chars]
            next_char = ''.join([indices_char[next_index] for next_index in next_indexes])

            generated += next_char
            sentence = sentence[3:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
