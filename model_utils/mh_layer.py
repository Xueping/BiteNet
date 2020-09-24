from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras


class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, direction, train, dropout, num_units, num_heads=10,  **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.direction = direction
        self.dropout = dropout
        self.train = train
        self.num_units = num_units
        self.q_linear = keras.layers.Dense(self.num_units, use_bias=False)
        self.k_linear = keras.layers.Dense(self.num_units, use_bias=False)
        self.v_linear = keras.layers.Dense(self.num_units, use_bias=False)

    def call(self, inputs):

        # because of self-attention, queries and keys is equal to inputs
        input_tensor, input_mask = inputs
        queries = input_tensor
        keys = input_tensor

        # Linear projections
        Q = self.q_linear(queries)  # (N, L_q, d)
        K = self.k_linear(keys)  # (N, L_k, d)
        V = self.v_linear(keys)  # (N, L_k, d)

        # print('Q shape: ', Q.get_shape())

        # Split and concat
        Q_ = tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0)  # (h*N, L_q, d/h)
        K_ = tf.concat(tf.split(K, self.num_heads, axis=2), axis=0)  # (h*N, L_k, d/h)
        V_ = tf.concat(tf.split(V, self.num_heads, axis=2), axis=0)  # (h*N, L_k, d/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, L_q, L_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5) # (h*N, L_q, L_k)

        # Key Masking
        key_masks = tf.sign(tf.reduce_sum(tf.abs(K_), axis=-1))  # (h*N, T_k)
        key_masks = tf.expand_dims(key_masks, 1)  # (h*N, 1, T_k)
        key_masks = tf.tile(key_masks, [1, Q_.get_shape().as_list()[1], 1])  # (h*N, T_q, T_k)

        # Apply masks to outputs
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)  # exp mask
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)

        n_visits = input_tensor.get_shape()[1]
        sw_indices = tf.range(n_visits, dtype=tf.int32)
        sw_col, sw_row = tf.meshgrid(sw_indices, sw_indices)
        if self.direction == 'diag':
            # shape of (n_visits, n_visits)
            attention_mask = tf.cast(tf.linalg.diag(- tf.ones([n_visits], tf.int32)) + 1, tf.bool)
        elif self.direction == 'forward':
            attention_mask = tf.greater(sw_row, sw_col)  # shape of (n_visits, n_visits)
        else:
            attention_mask = tf.greater(sw_col, sw_row)  # shape of (n_visits, n_visits)
        adder = (1.0 - tf.cast(attention_mask, outputs.dtype)) * -10000.0
        outputs += adder

        # softmax
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.reduce_sum(tf.abs(Q_), axis=-1))  # (h*N, T_q)
        query_masks = tf.expand_dims(query_masks, -1)  # (h*N, T_q, 1)
        query_masks = tf.tile(query_masks, [1, 1, tf.shape(K_)[1]])  # (h*N, T_q, T_k)

        # Apply masks to outputs
        outputs = outputs * query_masks

        # Dropouts
        if self.train:
            outputs = tf.nn.dropout(outputs, rate=self.dropout)
        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=2)  # (N, L_q, d)

        # input padding
        val_mask = tf.expand_dims(input_mask, -1)
        outputs = tf.multiply(outputs, tf.cast(val_mask, tf.float32), name='mask_for_high_rank')

        return outputs