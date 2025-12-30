import os
from typing import List

import numpy as np
import tensorflow as tf

from ists_.model.model_cls import ISTForecastingCLS, ISTInterpolationCLS, ISTEncoderCLS
from ists_.model.model_attnpool import ISTEncoder, ISTInterpolationAttnPool, ISTForecastingAttnPool
from ists_.model.wrapper import TimingCallback


def lr_schedule_warmup_linear_decay(max_lr, warmup_epochs, total_epochs):
    def scheduler(epoch, lr):
        if epoch < warmup_epochs:
            lr = max_lr * ((epoch + 1) / warmup_epochs)
        else:
            decay_epochs = total_epochs - warmup_epochs
            lr = max_lr * ((total_epochs - epoch) / decay_epochs)
        return lr
    return scheduler


def lr_schedule_warmup_linear(max_lr, warmup_steps=4000):
    class WarmupLinearSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, max_lr, warmup_steps):
            super(WarmupLinearSchedule, self).__init__()
            self.max_lr = max_lr
            self.warmup_steps = warmup_steps

        def __call__(self, step):
            step = tf.cast(step, dtype=tf.float32)
            warmup_steps = tf.cast(self.warmup_steps, dtype=tf.float32)

            lr = tf.cond(
                step < warmup_steps,
                lambda: self.max_lr * (step / warmup_steps),
                lambda: self.max_lr
            )
            return lr
        
        def get_config(self):
            return {
                "max_lr": self.max_lr,
                "warmup_steps": self.warmup_steps
            }
    return WarmupLinearSchedule(max_lr, warmup_steps)


class FixedPeakSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, warmup_steps=4000):
        super().__init__()
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, dtype=tf.float32)

        # Standard Noam shape (warmup then inverse square root decay)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (warmup_steps ** -1.5)
        
        # Calculate the scale to hit exactly peak_lr at step == warmup_steps
        # The unscaled value at warmup is 1/sqrt(warmup_steps)
        # We want: scale * (1/sqrt(warmup)) = peak_lr
        # Therefore: scale = peak_lr * sqrt(warmup)
        scale = self.peak_lr * tf.math.sqrt(warmup_steps)

        return scale * tf.math.minimum(arg1, arg2)
    
    def get_config(self):
        return {
            "peak_lr": self.peak_lr,
            "warmup_steps": self.warmup_steps
        }


def ModelCheckpointCallback(checkpoint_path):
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',  # 'val_mse',
        save_best_only=True,
        save_weights_only=True,
        mode='min',
        verbose=1
    )


def EarlyStoppingCallback(patience, start_from_epoch=0, min_delta=0):
    return tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # 'val_mse',
        patience=patience,
        mode='min',
        verbose=1,
        restore_best_weights=True,
        start_from_epoch=start_from_epoch,
        min_delta=min_delta,
    )


def get_spatial_array(x: np.ndarray, spt: List[np.ndarray]) -> List[np.ndarray]:
    if len(spt) == 0:
        return [x]
    spt_x = [x] + spt
    return spt_x


def data_reshuffle(x, spt, exg, mask_id, x_aux_id, mask_aux_id, time_ids=None):
    spt, exg = get_spatial_array(x, spt), get_spatial_array(x, exg)  # (V, B, T, F)
    spt, exg = np.stack(spt, axis=1), np.stack(exg, axis=1)  # (B, V, T, F)
    spt_mask, exg_mask = spt[:, :, :, mask_id], exg[:, :, :, mask_id]
    spt_aux, exg_aux = spt[:, :, :, x_aux_id], exg[:, :, :, x_aux_id]
    spt_aux_mask, exg_aux_mask = spt[:, :, :, mask_aux_id], exg[:, :, :, mask_aux_id]
    to_delete = [mask_id, x_aux_id, mask_aux_id]
    spt, exg = np.delete(spt, to_delete, axis=-1), np.delete(exg, to_delete, axis=-1)
    if time_ids is not None:
        spt_time, exg_time = spt[:, :, :, time_ids], exg[:, :, :, time_ids]
        spt_aux = np.concatenate([spt_aux[..., np.newaxis], spt_time], axis=-1)
        exg_aux = np.concatenate([exg_aux[..., np.newaxis], exg_time], axis=-1)
    return exg, exg_mask, exg_aux, exg_aux_mask, spt, spt_mask, spt_aux, spt_aux_mask


class ModelWrapper:
    def __init__(
            self,
            checkpoint_dir: str,
            model_params: dict,
            model_type: str,
            loss: str = 'mse',
            lr: float = 0.001,
            dev = False,
            *args, **kwargs
    ):
        self.checkpoint_dir = checkpoint_dir
        self.model_params = model_params
        self.loss = loss
        self.lr = lr
        # self.dev = dev
        self.dev = False

        self.null_id = np.where(np.array(self.model_params['feature_mask']) == 1)[0][0]
        self.model_params['feature_mask'] = np.delete(self.model_params['feature_mask'], self.null_id)

        self.model = None
        self.history = None
        self.epoch_times = dict()

        self.enc_cls, self.pretr_cls, self.finet_cls = {
            'istf_interp_cls': (ISTEncoderCLS, ISTInterpolationCLS, ISTForecastingCLS),
            'istf_attnpool': (ISTEncoder, ISTInterpolationAttnPool, ISTForecastingAttnPool),
        }[model_type]

    def load_pretrained_checkpoint(self, path: str):
        if os.path.exists(path):
            print(f'Loading pretrained model from {path}')
            self.model.encoder.load_weights(path)
            self.model.encoder.trainable = False
        pretr_mean_head_path = os.path.join(self.checkpoint_dir, 'pretr_mean_head_weights.npz')
        if os.path.exists(pretr_mean_head_path) and hasattr(self.model, 'mean_head'):
            weights = np.load(pretr_mean_head_path)
            weights = [weights[f] for f in weights]
            self.model.mean_head.set_weights(weights)
            self.model.mean_head.trainable = False
        pretr_mean_pool_path = os.path.join(self.checkpoint_dir, 'pretr_mean_pool_weights.npz')
        if os.path.exists(pretr_mean_pool_path) and hasattr(self.model, 'mean_pool'):
            weights = np.load(pretr_mean_pool_path)
            weights = [weights[f] for f in weights]
            self.model.mean_pool.set_weights(weights)
            self.model.mean_pool.trainable = False
        if os.path.exists(os.path.join(self.checkpoint_dir, 'pretr_intp_head_weights.npz')) and hasattr(self.model, 'intp_head'):
            weights = np.load(os.path.join(self.checkpoint_dir, 'pretr_intp_head_weights.npz'))
            weights = [weights[f] for f in weights]
            self.model.intp_head.set_weights(weights)
            self.model.intp_head.trainable = False

    def pretrain(
            self,
            x: np.ndarray,
            spt: List[np.ndarray],
            exg: List[np.ndarray],
            epochs: int = 50,
            batch_size: int = 32,
            verbose: int = 0,
            val_x: np.ndarray = None, val_spt: List[np.ndarray] = None, val_exg: List[np.ndarray] = None,
            early_stop_patience: int = None,
            exg_static=None, spt_static=None, val_exg_static=None, val_spt_static=None,
            **kwargs
    ):
        null_id = self.null_id
        time_ids = np.where(np.array(self.model_params['feature_mask']) == 2)[0]
        exg_static = exg_static if exg_static is not None else np.zeros((x.shape[0], 1, 0))
        spt_static = spt_static if spt_static is not None else np.zeros((x.shape[0], 1, 0))
        val_exg_static = val_exg_static if val_exg_static is not None else np.zeros((val_x.shape[0], 1, 0))
        val_spt_static = val_spt_static if val_spt_static is not None else np.zeros((val_x.shape[0], 1, 0))

        exg, exg_mask, exg_aux, exg_aux_mask, spt, spt_mask, spt_aux, spt_aux_mask = data_reshuffle(
            x, spt, exg, null_id, -2, -1, time_ids)
        # X = ((exg, exg_mask, exg_aux, exg_aux_mask), (spt, spt_mask, spt_aux, spt_aux_mask))
        X = ((exg, exg_mask, exg_static, exg_aux, exg_aux_mask), (spt, spt_mask, spt_static, spt_aux, spt_aux_mask))

        val_exg, val_exg_mask, val_exg_aux, val_exg_aux_mask, val_spt, val_spt_mask, val_spt_aux, val_spt_aux_mask = (
            data_reshuffle(val_x, val_spt, val_exg, null_id, -2, -1, time_ids))
        val_data = ((
                (val_exg, val_exg_mask, val_exg_static, val_exg_aux, val_exg_aux_mask),
                (val_spt, val_spt_mask, val_spt_static, val_spt_aux, val_spt_aux_mask)
        ), None)

        lr = self.lr
        lr = lr_schedule_warmup_linear(lr, warmup_steps=500)

        checkpoint_path = os.path.join(self.checkpoint_dir, 'cp.weights.h5')
        model_checkpoint = ModelCheckpointCallback(checkpoint_path)
        timing_callback = TimingCallback()
        callbacks = [model_checkpoint, timing_callback]
        warmup_epochs = 0  # max(1, int(epochs * 0.1))
        """lr_schedule = tf.keras.callbacks.LearningRateScheduler(
            lr_schedule_warmup_linear_decay(lr, warmup_epochs, epochs),
            verbose=1  # Set to 1 to log LR changes at each epoch
        )
        callbacks.append(lr_schedule)"""
        if early_stop_patience:
            early_stopping = EarlyStoppingCallback(
                early_stop_patience,
                start_from_epoch=warmup_epochs,
                min_delta=0
            )
            callbacks.append(early_stopping)

        encoder = self.enc_cls(**self.model_params)
        self.model = self.pretr_cls(encoder)

        X_dummy = tuple(tuple(np.zeros_like(x1[:batch_size]) for x1 in x) for x in X)
        self.model(X_dummy)
        self.model.summary(expand_nested=True)
        # return
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, global_clipnorm=1.0)
        self.model.compile(
            optimizer=optimizer,
            run_eagerly=self.dev,
        )

        self.history = self.model.fit(
            x=X,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_data,
            verbose=verbose,
            callbacks=callbacks
        )
        self.model.summary(expand_nested=True)
        print("Variable embeddings scale:", self.model.encoder.ve_scale.numpy().item())
        self.epoch_times['pretr'] = timing_callback.epoch_times

        # Load best model
        self.model.load_weights(checkpoint_path)
        self.model.encoder.save_weights(self.checkpoint_dir + '/pretr_encoder.weights.h5')
        if hasattr(self.model, 'mean_head'):
            weights = self.model.mean_head.get_weights()
            np.savez(self.checkpoint_dir + '/pretr_mean_head_weights.npz', *weights)
        if hasattr(self.model, 'mean_pool'):
            weights = self.model.mean_pool.get_weights()
            np.savez(self.checkpoint_dir + '/pretr_mean_pool_weights.npz', *weights)
        if hasattr(self.model, 'intp_head'):
            weights = self.model.intp_head.get_weights()
            np.savez(self.checkpoint_dir + '/pretr_intp_head_weights.npz', *weights)
        os.remove(checkpoint_path)

    def fit(
            self,
            x: np.ndarray,
            spt: List[np.ndarray],
            exg: List[np.ndarray],
            y: np.ndarray,
            epochs: int = 50,
            batch_size: int = 32,
            verbose: int = 0,
            val_x: np.ndarray = None, val_spt: List[np.ndarray] = None, val_exg: List[np.ndarray] = None,
            val_y: np.ndarray = None,
            early_stop_patience: int = None,
            exg_static=None, spt_static=None, val_exg_static=None, val_spt_static=None,
    ):
        null_id = self.null_id
        exg_static = exg_static if exg_static is not None else np.zeros((x.shape[0], 1, 0))
        spt_static = spt_static if spt_static is not None else np.zeros((x.shape[0], 1, 0))
        val_exg_static = val_exg_static if val_exg_static is not None else np.zeros((val_x.shape[0], 1, 0))
        val_spt_static = val_spt_static if val_spt_static is not None else np.zeros((val_x.shape[0], 1, 0))

        exg, exg_mask, _, _, spt, spt_mask, _, _ = data_reshuffle(x, spt, exg, null_id, -2, -1)
        X = ((exg, exg_mask, exg_static), (spt, spt_mask, spt_static))

        val_exg, val_exg_mask, _, _, val_spt, val_spt_mask, _, _ = data_reshuffle(
            val_x, val_spt, val_exg, null_id, -2, -1
        )
        val_X = ((val_exg, val_exg_mask, val_exg_static), (val_spt, val_spt_mask, val_spt_static))

        # print('\n## Warming-up the task-specific head ##\n')

        lr = self.lr
        lr = lr_schedule_warmup_linear(lr, warmup_steps=500)

        # self.model_params['last_layer_no_mask'] = True  # no mask in the last layer
        lr = 1e-4  # reduce learning rate for warmup and finetuning
        lr = lr_schedule_warmup_linear(lr, warmup_steps=500)

        encoder = self.enc_cls(**self.model_params)
        self.model = self.finet_cls(encoder)
        # optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        optimizer = {"optimizer": tf.keras.optimizers.Adam(learning_rate=lr, global_clipnorm=1.0)}
        """optimizer = {
            "encoder_optimizer": tf.keras.optimizers.Adam(learning_rate=lr),  # frozen encoder, unused
            "head_optimizer": tf.keras.optimizers.Adam(learning_rate=lr)
        }"""

        X_dummy = tuple(tuple(np.zeros_like(x1[:batch_size]) for x1 in x) for x in X)
        self.model(X_dummy)

        pretrained_path = os.path.join(self.checkpoint_dir, 'pretr_encoder.weights.h5')
        self.load_pretrained_checkpoint(pretrained_path)
        '''self.model.summary(expand_nested=True)  # COMMENT FROM HERE FOR NO WARMUP

        self.model.compile(loss=self.loss, **optimizer, metrics=['mae', 'mse'], run_eagerly=self.dev)

        checkpoint_path = os.path.join(self.checkpoint_dir, 'cp.weights.h5')
        model_checkpoint = ModelCheckpointCallback(checkpoint_path)
        timing_callback = TimingCallback()
        callbacks = [model_checkpoint, timing_callback]
        early_stopping = EarlyStoppingCallback(2, min_delta=0)
        callbacks.append(early_stopping)

        warmup_epochs = max(1, int(epochs * 0.1))
        epochs = epochs - warmup_epochs

        self.history = self.model.fit(
            x=tuple(X),
            y=y,
            epochs=warmup_epochs,
            batch_size=batch_size,
            validation_data=(val_X, val_y),
            verbose=verbose,
            callbacks=callbacks
        )
        self.model.summary(expand_nested=True)
        self.epoch_times['finet1'] = timing_callback.epoch_times

        # Load best model
        self.model.load_weights(checkpoint_path)
        self.model.save_weights(self.checkpoint_dir + '/finet1.weights.h5')
        os.remove(checkpoint_path)  # COMMENT UNTIL HERE FOR NO WARMUP'''

        print('\n## Finetuning the encoder ##\n')

        """encoder = ISTEncoderCLS(**self.model_params)
        self.model = ISTForecastingCLS(encoder)
        self.model(X_dummy)
        self.model.load_weights(self.checkpoint_dir + '/finet1.weights.h5')"""

        self.model.encoder.trainable = True
        if hasattr(self.model.encoder, 'cls_token'): self.model.encoder.cls_token.trainable = True
        if hasattr(self.model, 'mean_head'): self.model.mean_head.trainable = True
        if hasattr(self.model, 'mean_pool'): self.model.mean_pool.trainable = True
        # if hasattr(self.model.encoder, 'variable_embeddings'): self.model.encoder.variable_embeddings.trainable = False
        for emb in self.model.encoder.embedder.time_embedders:
            emb.trainable = False  # fixed embeddings
        self.model.summary(expand_nested=True)

        optimizer = {"optimizer": tf.keras.optimizers.Adam(learning_rate=lr, global_clipnorm=1.0)}
        """optimizer = {
            "encoder_optimizer": tf.keras.optimizers.Adam(learning_rate=1e-4),
            "head_optimizer": tf.keras.optimizers.Adam(learning_rate=lr)
        }"""
        self.model.compile(loss=self.loss, **optimizer, metrics=['mae', 'mse'], run_eagerly=self.dev)

        # reinitialize callbacks
        checkpoint_path = os.path.join(self.checkpoint_dir, 'cp.weights.h5')
        model_checkpoint = ModelCheckpointCallback(checkpoint_path)
        timing_callback = TimingCallback()
        callbacks = [model_checkpoint, timing_callback]
        warmup_epochs = 0  # max(1, int(epochs * 0.1))
        """lr_schedule = tf.keras.callbacks.LearningRateScheduler(
            lr_schedule_warmup_linear_decay(lr, warmup_epochs, epochs),
            verbose=1  # Set to 1 to log LR changes at each epoch
        )
        callbacks.append(lr_schedule)"""
        if early_stop_patience:
            early_stopping = EarlyStoppingCallback(
                early_stop_patience,
                start_from_epoch=warmup_epochs,
                min_delta=0
            )
            callbacks.append(early_stopping)

        self.history = self.model.fit(
            x=tuple(X),
            y=y,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(val_X, val_y),
            verbose=verbose,
            callbacks=callbacks
        )
        self.model.summary(expand_nested=True)
        print("Variable embeddings scale:", self.model.encoder.ve_scale.numpy().item())
        self.epoch_times['finet2'] = timing_callback.epoch_times

        # Load best model
        self.model.load_weights(checkpoint_path)
        self.model.save(self.checkpoint_dir + '/model_finet2.keras')
        os.remove(checkpoint_path)

    def predict(
            self, x: np.ndarray,
            spt: List[np.ndarray],
            exg: List[np.ndarray],
            exg_static=None, spt_static=None,
    ):
        exg_static = exg_static if exg_static is not None else np.zeros((x.shape[0], 1, 0))
        spt_static = spt_static if spt_static is not None else np.zeros((x.shape[0], 1, 0))

        exg, exg_mask, _, _, spt, spt_mask, _, _ = data_reshuffle(x, spt, exg, self.null_id, -2, -1)
        X = ((exg, exg_mask, exg_static), (spt, spt_mask, spt_static))

        y_preds = self.model.predict(X)

        return y_preds

    def pretrain_predict(
            self, x: np.ndarray,
            spt: List[np.ndarray],
            exg: List[np.ndarray],
            exg_static=None, spt_static=None,
    ):
        exg_static = exg_static if exg_static is not None else np.zeros((x.shape[0], 1, 0))
        spt_static = spt_static if spt_static is not None else np.zeros((x.shape[0], 1, 0))

        null_id = self.null_id
        time_ids = np.where(np.array(self.model_params['feature_mask']) == 2)[0]

        exg, exg_mask, exg_aux, exg_aux_mask, spt, spt_mask, spt_aux, spt_aux_mask = data_reshuffle(
            x, spt, exg, null_id, -2, -1, time_ids)
        X = ((exg, exg_mask, exg_static, exg_aux, exg_aux_mask), (spt, spt_mask, spt_static, spt_aux, spt_aux_mask))

        y_pred = self.model.predict(X)

        y_true = exg[:, :, :, 0]  # FIXME: generalize
        y_fill = exg_aux[:, :, :, 0]

        return y_pred, y_true, y_fill, exg_mask, exg_aux_mask
