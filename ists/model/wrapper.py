from typing import TypeVar

import numpy as np
import tensorflow as tf

from .model import STTransformer
from ..preprocessing import StandardScalerBatch, MinMaxScalerBatch

T = TypeVar('T', bound=tf.keras.Model)


def get_transformer(transform_type: str) -> object:
    # Return the selected model
    if transform_type == 'standard':
        return StandardScalerBatch(p1=1, p2=99)
    elif transform_type == 'minmax':
        return MinMaxScalerBatch()
    else:
        raise ValueError('Transformer {} is not supported, it must be "standard" or "minmax"')


def get_model(model_type: str, model_params) -> T:
    # Return the selected model
    if model_type == 'sttransformer':
        return STTransformer(**model_params)
    else:
        raise ValueError('Model {} is not supported, it must be "sttransformer"')


class ModelWrapper(object):
    def __init__(self,
                 model_type: str,
                 model_params: dict,
                 transform_type=None,
                 loss: str = 'mse',
                 lr: float = 0.001,
                 ):
        self.model = get_model(model_type, model_params)
        self.transform_type = transform_type
        if transform_type:
            self.transformer = get_transformer(transform_type)
            self.spt_transformer = get_transformer(transform_type)
            self.exg_transformer = get_transformer(transform_type)

        self.loss = loss
        self.lr = lr
        self.history = None

        self.feature_mask = model_params['feature_mask']
        self.exg_feature_mask = model_params['exg_feature_mask']

    def _fit_transform(self, x: np.ndarray, spt: np.ndarray, exg: np.ndarray):
        if self.transform_type:
            cond_x = np.array(self.feature_mask) == 0
            x[:, :, cond_x] = self.transformer.fit_transform(x[:, :, cond_x])

            spt_size = len(spt)
            spt_all = np.concatenate(spt, axis=1)
            spt_all[:, :, cond_x] = self.spt_transformer.fit_transform(spt_all[:, :, cond_x])
            spt = np.split(spt_all, spt_size, axis=1)

            cond_exg = np.array(self.exg_feature_mask) == 0
            exg[:, :, cond_exg] = self.exg_transformer.fit_transform(exg[:, :, cond_exg])
        return x, spt, exg

    def _label_transform(self, y):
        if self.transform_type:
            y = self.transformer.transform(y)

        return y

    def _label_inverse_transform(self, y):
        if self.transform_type:
            y = self.transformer.inverse_transform(y)
        return y

    def fit(
            self,
            x: np.ndarray,
            spt: np.ndarray,
            exg: np.ndarray,
            y: np.ndarray,
            epochs: int = 50,
            batch_size: int = 32,
            validation_split: float = 0.1,
            verbose: int = 0
    ):
        x, spt, exg = self._fit_transform(x, spt, exg)
        y = self._label_transform(y)

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)  # , clipnorm=1.0, clipvalue=0.5)
        self.model.compile(
            loss=self.loss,
            optimizer=optimizer,
            metrics=['mae', 'mse'],
            # run_eagerly=True,
        )

        self.history = self.model.fit(
            x=[x] + [exg] + spt,
            y=y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )

    def predict(self, x: np.ndarray, spt: np.ndarray, exg: np.ndarray):
        x, spt, exg = self._fit_transform(x, spt, exg)

        y_preds = self.model.predict([x] + [spt] + [exg])

        y_preds = self._label_inverse_transform(y_preds)

        return y_preds
