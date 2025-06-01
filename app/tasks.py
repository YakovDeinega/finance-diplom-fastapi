from pathlib import Path

import pandas as pd
from celery.utils.log import get_task_logger

from moexalgo import session, Ticker


import tensorflow as tf

from app.config import settings

from app.common_utils import create_df, create_dataset, create_model

from .celery import celery_app

logger = get_task_logger(__name__)


@celery_app.task(name='train_model', queue='fastapi_queue')
def train_model(ticker):
    try:
        logger.info(f"Starting training for {ticker}")

        model_dir = Path('/app/keras_models')
        model_dir.mkdir(exist_ok=True, parents=True)
        model_path = model_dir / f'{ticker}_model.keras'

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
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=str(model_path), save_best_only=True)

        # Создание и обучение модели
        model_lstm = create_model((X_train.shape[1], X_train.shape[2]))
        hist = model_lstm.fit(X_train, y_train, epochs=20, batch_size=128, callbacks=[early_stopping, model_checkpoint],
                              validation_data=(X_test, y_test))
        model_lstm.save(str(model_path))
        logger.info(f"Model saved to {model_path}")
    except Exception as exc:
        logger.error(f"Error in train_model: {str(exc)}")
        raise exc
