import numpy as np
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential

np.random.seed(1337)  # reproduce results

time_series_length = 1000

# random time series.
ts = np.random.standard_normal(size=time_series_length)

m = Sequential()
m.add(LSTM(10, batch_input_shape=(1, 1, 1), stateful=True))
m.add(Dense(1))

m.compile(optimizer='adam', loss='mse')

print(m.summary())

# ts = [0, 1, 2, 3, 4]

# x = [0, 1, 2, 3]
#      |  |  |  |
# y = [1, 2, 3, 4]
x = np.reshape(ts[:-1], (-1, 1, 1))
y = np.reshape(ts[1:], (-1, 1))

for i in range(len(x)):
    loss = m.train_on_batch(x[i:i + 1], y[i:i + 1])
    print('Training for step = {}, loss = {}'.format(i, loss))
