import tensorflow as tf

from ists.model.encoder import GlobalSelfAttention, CrossAttention, FeedForward, EncoderLayer


class ContextualEncoderLayer(tf.keras.layers.Layer):

    def __init__(self, *, d_model, num_heads, dff, activation='relu', dropout_rate=0.1, l2_reg=None):
        super().__init__()

        reg = {}
        if l2_reg:
            reg['kernel_regularizer'] = tf.keras.regularizers.l2(l2_reg)

        self.loc_attn = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate,
            **reg
        )

        self.glb_attn = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate,
            **reg
        )

        self.cross_attn = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate,
            **reg
        )

        self.ffn = FeedForward(
            d_model=d_model,
            dff=dff,
            activation=activation,
            dropout_rate=dropout_rate, **reg
        )

    def call(self, x, context):  # x: (b, t, e) context: (v, b, t, e)
        context_shape = tf.shape(context)
        v, b, t, e = context_shape[0], context_shape[1], context_shape[2], context_shape[3]

        x = self.loc_attn(x)

        context = tf.reshape(context, (-1, t, e))  # context: (v*b, t, e)
        context = self.glb_attn(context)

        context = tf.reshape(context, (v, -1, t, e))  # context: (v, b, t, e)
        context = tf.transpose(context, perm=[1, 0, 2, 3])  # context: (b, v, t, e)
        context = tf.reshape(context, (b, -1, e))  # context: (b, v*t, e)
        context = self.glb_attn(context)

        x = self.cross_attn(x, context)

        x_context = tf.concat([x, context], axis=1)  # x_context: (b, (v+1)*t, e)
        x_context = self.ffn(x_context)

        x, context = tf.split(x_context, [t, v*t], axis=1)  # x: (b, t, e); context: (b, v*t, e)
        context = tf.reshape(context, (b, v, t, e))  # context: (b, v, t, e)
        context = tf.transpose(context, perm=[1, 0, 2, 3])  # context: (v, b, t, e)

        return x, context


class ContextualEncoder(tf.keras.layers.Layer):

    def __init__(self, *, d_model, num_heads, dff, activation='relu', num_layers=1, dropout_rate=0.1, l2_reg=None,
                 do_exg=True, do_spt=True, force_target=True, **kwargs):
        super().__init__()
        self.num_layers = num_layers

        layer_cls = ContextualEncoderLayer
        if 'layer_cls' in kwargs and kwargs['layer_cls']:
            layer_cls = kwargs['layer_cls']

        self.exg_encs = [layer_cls(
            d_model=d_model, num_heads=num_heads, dff=dff, activation=activation, dropout_rate=dropout_rate, l2_reg=l2_reg
        ) for _ in range(num_layers)] if do_exg or force_target else [lambda x: x for _ in range(self.num_layers)]

        self.spt_encs = [layer_cls(
            d_model=d_model, num_heads=num_heads, dff=dff, activation=activation, dropout_rate=dropout_rate, l2_reg=l2_reg
        ) for _ in range(num_layers)] if do_spt or force_target else [lambda x: x for _ in range(self.num_layers)]

    def call(self, inputs):
        x, exg_ctx, spt_ctx = inputs
        for i in range(self.num_layers):
            x, exg_ctx = self.exg_encs[i](x, exg_ctx)
            x, spt_ctx = self.spt_encs[i](x, spt_ctx)
        return x, exg_ctx, spt_ctx


class GlobalEncoderLayer(tf.keras.layers.Layer):

    def __init__(self, *, d_model, num_heads, dff, activation='relu', num_layers=1, dropout_rate=0.1, l2_reg=None,
                 do_exg=True, do_spt=True, do_glb=True, force_target=True, **kwargs
                 ):
        super(GlobalEncoderLayer, self).__init__()

        self.num_layers = num_layers

        self.spatial_encoders = [
            EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dff=dff,
                dropout_rate=dropout_rate,
                activation=activation,
                l2_reg=l2_reg
            )
            for _ in range(self.num_layers)
        ] if do_spt or force_target else [lambda x: x for _ in range(self.num_layers)]

        self.exogenous_encoders = [
            EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dff=dff,
                dropout_rate=dropout_rate,
                activation=activation,
                l2_reg=l2_reg
            )
            for _ in range(self.num_layers)
        ] if do_exg or force_target else [lambda x: x for _ in range(self.num_layers)]

        if force_target and not do_exg and not do_spt:
            self.exogenous_encoders = [lambda x: x for _ in range(self.num_layers)]

        self.global_encoders = [
            EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dff=dff,
                dropout_rate=dropout_rate,
                activation=activation,
                l2_reg=l2_reg
            )
            for _ in range(self.num_layers)
        ] if do_glb else [lambda x: x for _ in range(self.num_layers)]

        # self.last_attn_scores = None

    def call(self, x, **kwargs):
        exg_x, spt_x = x

        for i in range(self.num_layers):
            exg_x = self.exogenous_encoders[i](exg_x)
            spt_x = self.spatial_encoders[i](spt_x)

            x = tf.concat([exg_x, spt_x], axis=1)
            x = self.global_encoders[i](x)
            exg_x, spt_x = tf.split(x, [exg_x.shape[1], spt_x.shape[1]], axis=1)

        return exg_x, spt_x
