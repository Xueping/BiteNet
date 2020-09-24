import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops import math_ops
from model_utils import ap_layer, embedding_layer, common_layer,\
    position_encoding_layer as pos_layer
from model_utils.attentionLayers import Flatten
from utils.configs import cfg
from utils.model_utils import Reshape


class BiteNet(object):

    def __init__(self, dataset):

        # ------ start ------
        self.lr = 0.0001
        self.dropout_rate = cfg.dropout
        self.n_intervals = dataset.days_size
        self.n_visits = cfg.valid_visits
        self.n_codes = dataset.max_len_visit
        self.vocabulary_size = len(dataset.dictionary)
        self.digit3_size = len(dataset.dictionary_3digit)
        self.pos_encoding = cfg.pos_encoding
        self.embedding_size = cfg.embedding_size
        self.num_hidden_layers = cfg.num_hidden_layers
        self.train = cfg.train
        self.predict_type = cfg.predict_type
        self.model = None
        self.version = cfg.version

        # ---- place holder -----
        # tensor with shape [batch_size, n_visits, n_codes]
        self.code_inputs = keras.layers.Input(shape=(self.n_visits, self.n_codes,), dtype=tf.int32, name='train_inputs')
        self.interval_inputs = keras.layers.Input(shape=(self.n_visits,), dtype=tf.int32, name='interval_inputs')

        # tensor with shape [batch_size, n_visits, n_codes]
        self.inputs_mask = math_ops.not_equal(self.code_inputs, 0)
        # tensor with shape [batch_size, n_visits]
        visit_mask = tf.reduce_sum(tf.cast(self.code_inputs, tf.int32), -1)
        self.visit_mask = tf.cast(visit_mask, tf.bool)

        self.embedding = embedding_layer.EmbeddingSharedWeights(self.vocabulary_size,
                                                                self.embedding_size,name='codes_embedding')
        self.interval_embedding = embedding_layer.EmbeddingSharedWeights(self.n_intervals, self.embedding_size,
                                                         name='interval_embedding')
        self.params = dict()
        self.params["hidden_size"] = self.embedding_size
        self.params["filter_size"] = self.embedding_size
        self.params["dropout"] = self.dropout_rate
        self.params["allow_ffn_pad"] = False
        self.params["num_hidden_layers"] = self.num_hidden_layers
        self.params["is_scale"] = False
        self.params["direction"] = 'diag'
        self.params["num_heads"] = cfg.num_heads

    def build_network(self):

        # embed inputs, tensor with shape [batch_size, n_visits, n_codes, embedding_size]
        e = self.embedding(self.code_inputs)

        # input tensor, reshape 4 dimension to 3
        # e = keras.layers.Reshape((self.n_codes, self.embedding_size), name='codes_reshape')(e)
        e = Flatten(2, name='code_flaten')(e)

        # input mask, reshape 3 dimension to 2
        e_mask = Flatten(1, name='mask_flaten')(self.inputs_mask)

        # only attention pooling
        # v = ap_layer.AttentionPooling(self.embedding_size, name='intra_attn_pool')((e, e_mask))

        # Vanilla Encoder
        h = common_layer.EncoderStack(self.params, self.train, self.version,
                                       self.embedding_size, name='Vanilla_encoder')((e, e_mask))
        # Attention pooling
        v = ap_layer.AttentionPooling(self.embedding_size, name='intra_attn_pool')((h,e_mask))
        # print(v.get_shape())
        # visit embedding, reshape 2 dimension to 3

        # v = keras.layers.Reshape((self.n_visits, self.embedding_size,))(v)
        v = Reshape()((v, self.code_inputs, self.embedding_size))
        # v = Reshape()((v, self.n_visits, self.embedding_size))
        print(v.get_shape())

        #-------------- interval encoding ------------------
        if cfg.pos_encoding == 'None':
            e_p = pos_layer.PositionEncoding(self.embedding_size, False)(self.interval_inputs)
        else:
            if cfg.pos_encoding == 'embedding':
                # position embedding strategy
                e_p = self.interval_embedding(self.interval_inputs)
            else:
                # position encoding strategy
                e_p = pos_layer.PositionEncoding(self.embedding_size, True)(self.interval_inputs)
        v = keras.layers.Add()([e_p, v])

        # forward mask in interVisit self-attention
        self.params["direction"] = 'forward'
        o_fw = common_layer.EncoderStack(self.params, self.train, self.version,self.embedding_size,
                                         name='forward_encoder')((v, self.visit_mask))
        # o_fw = keras.layers.Add()([e_p, o_fw])
        u_fw = ap_layer.AttentionPooling(self.embedding_size, name='forward_attn_pool')((o_fw, self.visit_mask))

        # backward mask in interVisit self-attention
        self.params["direction"] = 'backward'
        o_bw = common_layer.EncoderStack(self.params, self.train, self.version,self.embedding_size,
                                         name='backward_encoder')((v, self.visit_mask))
        # o_bw = keras.layers.Add()([e_p, o_bw])
        u_bw = ap_layer.AttentionPooling(self.embedding_size, name='backward_attn_pool')((o_bw, self.visit_mask))

        # concatenate outputs of forward and backward
        u_bi = keras.layers.Concatenate()([u_fw, u_bw])

        # feed forward network with two layers (embedding_size, embedding_size)
        s = keras.layers.Dense(self.embedding_size, 'relu')(u_bi)
        # s = keras.layers.Dense(self.embedding_size)(s)

        if self.predict_type == 'dx':
            s = keras.layers.Dropout(self.dropout_rate)(s)
            logits = keras.layers.Dense(self.digit3_size, activation='sigmoid')(s)
            self.model = keras.Model(inputs=[self.code_inputs, self.interval_inputs],
                                     outputs=logits, name='hierarchicalSA')
            self.model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
            # self.model.compile(optimizer=keras.optimizers.Adam(0.0001),
                               loss='binary_crossentropy',
                               metrics=['accuracy'])

        elif self.predict_type == 're':
            s = keras.layers.Dropout(self.dropout_rate)(s)
            logits = keras.layers.Dense(1, activation='sigmoid', use_bias=True)(s)
            self.model = keras.Model(inputs=[self.code_inputs, self.interval_inputs],
                                     outputs=logits, name='BiteNet')
            self.model.compile(optimizer=keras.optimizers.Adam(0.001),
            # self.model.compile(optimizer=keras.optimizers.RMSprop(0.0001),
                               loss=keras.losses.BinaryCrossentropy(),
                               metrics=['accuracy'])

        # multiple tasks
        else:
            s = keras.layers.Dropout(self.dropout_rate)(s)
            logits_dx = keras.layers.Dense(self.digit3_size, activation='sigmoid', name='los_dx')(s)
            logits_re = keras.layers.Dense(1, activation='sigmoid', use_bias=True, name='los_re')(s)
            self.model = keras.Model(inputs=[self.code_inputs, self.interval_inputs],
                                     outputs=[logits_dx, logits_re],
                                     name='hierarchicalSA')
            self.model.compile(keras.optimizers.RMSprop(learning_rate=0.001),
                               loss={'los_dx': 'binary_crossentropy',
                                     'los_re': 'binary_crossentropy'},
                               loss_weights={'los_dx': 1., 'los_re': 0.1},
                               metrics={'los_dx': 'accuracy', 'los_re': 'accuracy'})

        print(self.model.summary())

