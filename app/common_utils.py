from pathlib import Path

import numpy as np
import pandas as pd
from datetime import date
from moexalgo import session, Ticker
from keras import layers
import keras

import tensorflow as tf

from app.config import settings


def create_df(stock, per):
    s = stock.candles(start='2012-01-01', end='2013-01-01', period=per)
    for i in range(1, 14):
        tmp = stock.candles(start=str(date(2012+i, 1, 1)), end=str(date(2013+i, 1, 1)), period=per)
        s = pd.concat([s, tmp], ignore_index=True)

    return s


def create_model(input_shape):
    model = keras.Sequential()
    model.add(layers.LSTM(64, activation='gelu', return_sequences=True, input_shape=input_shape))
    model.add(layers.Dropout(0))
    model.add(layers.LSTM(32, activation='gelu'))
    model.add(layers.Dropout(0))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mae')
    return model


def create_dataset(data, time_steps=20):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    print(np.array(X))
    print(np.array(y))
    return np.array(X), np.array(y)


def train_model(ticker):
    # Подготовка данных
    session.TOKEN = settings.MOEX_SERVICE_TOKEN
    s = create_df(Ticker(ticker), '1h')
    tmp = s.copy()
    tmp.index = pd.to_datetime(tmp.end)
    tmp = tmp['close']
    X, y = create_dataset(tmp)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    X_train, y_train = X[:30000], y[:30000]
    X_test, y_test = X[30000:], y[30000:]

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=f'{ticker}_model.keras', save_best_only=True)

    # Создание и обучение модели
    model_lstm = create_model((X_train.shape[1], X_train.shape[2]))
    hist = model_lstm.fit(X_train, y_train, epochs=20, batch_size=128, callbacks=[early_stopping, model_checkpoint],
                          validation_data=(X_test, y_test))


def predict_cost(data, ticker):
    model_path = Path(f'{ticker}_model.keras')
    if not model_path.exists():
        train_model(ticker)

    loaded_model = keras.models.load_model(str(model_path))

    X_test, y_test = create_dataset(data)
    tempor = X_test.copy()
    per = y_test.copy()
    predict = loaded_model.predict(tempor)
    x_t = tempor.copy()

    for i in range(20):
        x_t = np.array([np.append(x_t, per)[1:]])
        per = loaded_model.predict(x_t)
        predict = np.append(predict, per)

    return predict.tolist()
