import datetime
import numpy as np
import pandas as pd
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense

TRAIN_END = np.datetime64('2010-12-31')
CLOSE = 'Adj Close'
RETURNS = 'Returns'
NEXT_DAY_RETURNS = 'Next Day Returns'

# The file snp.csv contains Yahoo Finance historical data from GSPC to
# 2016-12-15 inclusive.
snp = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "snp.csv"),
    index_col='Date',
    parse_dates=['Date'])
snp.sort_index(inplace=True)

# Add the log returns then drop the first day for which we can't know the
# returns on that day as we don't know the previous close.
returns = np.diff(np.log(snp[CLOSE].values))
snp = snp.iloc[1:]
snp.loc[:, RETURNS] = returns

# Add the value to be predicted, i.e. the next day's returns. We don't know
# the next day's return for the lat day so we drop it.
snp.loc[:, NEXT_DAY_RETURNS] = snp[RETURNS].shift(-1)
snp = snp.iloc[:-1]

assert np.abs(
    snp.iloc[1][RETURNS] - np.log(snp.iloc[1][CLOSE] / snp.iloc[0][CLOSE])
  < 1e-12)

snp_train = snp.loc[:TRAIN_END]
snp_test = snp.loc[TRAIN_END + np.timedelta64(1, 'D'):]

assert len(snp_train) + len(snp_test) == len(snp)

# Single LSTM layer and Dense layer for regression.
model = Sequential([
        LSTM(4, input_dim=1),
        Dense(1)])
model.compile(loss='mean_squared_error', optimizer='adam')

# Shape the X, y inputs into format expected by LTSM.
X_train = snp_train[[RETURNS]].values
X_train = X_train.reshape((-1, 1, X_train.shape[1]))
y_train = snp_train[NEXT_DAY_RETURNS].values
model.fit(X_train, y_train, nb_epoch=10, batch_size=1)

print("\nEvaluation...")

mmse = np.mean((snp_test[RETURNS] - snp_test[RETURNS].mean())**2)
print("    Mean return prediction: {mse}".format(mse=mmse))

X_test = snp_test[[RETURNS]].values
X_test = X_test.reshape((-1, 1, X_test.shape[1]))
y_test = snp_test[NEXT_DAY_RETURNS].values
nnmse = model.evaluate(X_test, y_test)
print("    NN MSE {mse}".format(mse=nnmse))

