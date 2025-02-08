import numpy as np
import tensorflow as tf

from ists.model.embedding import TemporalEmbedding
from ists.model.model import FinalLayersGRU


class EmbGRUModel(tf.keras.Model):

    def __init__(self, *,
                 feature_mask,
                 kernel_size,
                 d_model,
                 fff,
                 dropout_rate=0.1,
                 l2_reg=None,
                 do_exg=True, do_spt=True, do_emb=True,
                 **kwargs):
        super().__init__()
        feature_mask = np.array(feature_mask)
        # assuming features mask for a univariate series with one or more temporal features: [0, 1, 2...]
        arg_feat, arg_null, arg_time = (feature_mask == 0), (feature_mask == 1), (feature_mask == 2)
        feature_mask_range = np.arange(len(feature_mask))
        self.time_features = kwargs["time_features"]  # todo: adjust
        self.null_id = feature_mask_range[arg_null][0] if arg_null.any() else None
        self.feat_id = feature_mask_range[arg_feat][0]
        self.time_ids = feature_mask_range[arg_time]
        self.do_exg, self.do_spt = do_exg, do_spt
        n_features = 1
        if do_exg:
            n_features += kwargs['exg_size'] - 1
        if do_spt:
            n_features += kwargs['spatial_size'] - 1
        feature_mask = [0] * n_features
        if self.null_id is not None:
            feature_mask += [1] * kwargs['exg_size']
        feature_mask += [2] * sum(arg_time)
        feature_mask = np.array(feature_mask)

        if do_emb:
            feature_mask[feature_mask == 1] = 0  # consider the null indicator as a feature
            self.emb = TemporalEmbedding(
                d_model=d_model,
                kernel_size=kernel_size,
                feature_mask=feature_mask,
                time_features=self.time_features,
            )
            self.dropout = tf.keras.layers.Dropout(dropout_rate)

        else:
            feature_mask_selector = np.isin(feature_mask, [0, 1])
            for i in range(len(feature_mask_selector)):
                if not feature_mask_selector[i]:
                    feature_mask_selector[i] = True
                    break
            feature_mask = feature_mask[feature_mask_selector]
            # the difference with mTAN input is that I don't set missing values to 0
            arg_time = (feature_mask == 2)
            if arg_time.any():
                def norm_time(x):
                    T = tf.shape(x)[1]
                    t = tf.linspace(0.0, 1.0, T)
                    t = tf.reshape(t, (1, -1, 1))
                    x = tf.boolean_mask(x, feature_mask_selector, axis=2)
                    x = tf.where(arg_time, t, x)
                    return x

                self.emb = tf.keras.layers.Lambda(norm_time)
            else:
                self.emb = tf.keras.layers.Lambda(lambda x: x)
            self.dropout = tf.keras.layers.Lambda(lambda x: x)

        self.final_layers = FinalLayersGRU(fff, dropout_rate, l2_reg)

    def split_data_attn_mask(self, x):
        if self.null_id is None:
            return x, None
        ids_to_keep = [*range(x.shape[-1])]
        ids_to_keep.remove(self.null_id)
        return tf.gather(x, ids_to_keep, axis=-1), 1 - tf.gather(x, self.null_id, axis=-1)

    def call(self, inputs, **kwargs):
        exg_x, spt_x = inputs[0], inputs[1]  # (b, v, t, f)
        x, exg_x, spt_x = exg_x[:, 0:1], exg_x[:, 1:], spt_x[:, 1:]
        if self.do_exg:
            x = tf.concat([x, exg_x], axis=1)
        if self.do_spt:
            x = tf.concat([x, spt_x], axis=1)
        x = tf.transpose(x, (0, 2, 1, 3))  # (b, t, v, f)

        t = tf.gather(x[:, :, 0], self.time_ids, axis=-1)  # (b, t, 1)
        if self.null_id is None:
            m = tf.zeros_like(x)[:, :, 0:0, 0]  # (b, t, 0)
        else:
            m = 1 - tf.gather(x, [self.null_id], axis=-1)  # (b, t, v, 1)
            m = tf.squeeze(m, axis=-1)  # (b, t, v)
        x = tf.gather(x, [self.feat_id], axis=-1)  # (b, t, v, 1)
        x = tf.squeeze(x, axis=-1)  # (b, t, v)
        x = tf.concat([x, m, t], axis=-1)  # (b, t, v + 1) if null_id is None else (b, t, v*2 + 1)

        x = self.emb(x)
        x = self.dropout(x)
        pred = self.final_layers(x)

        return pred
