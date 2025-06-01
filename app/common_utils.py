from pathlib import Path

import numpy as np
import pandas as pd
from datetime import date
from keras import layers
import keras


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
    return np.array(X), np.array(y)


def predict_cost(data, ticker):
    from app.tasks import train_model

    model_path = Path(f'keras_models/{ticker}_model.keras')
    if not model_path.exists():
        train_model.delay(ticker)
        return []

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
