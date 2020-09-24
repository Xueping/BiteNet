import tensorflow as tf
from tensorflow import keras
from model_utils.normalization_layer import LayerNormalization
from model_utils.mh_layer import MultiHeadAttention
from model_utils.ffn_layer import FeedForwardNetwork
from model_utils.attentionLayers import Flatten, Reconstruct
VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER

@tf.function
def exp_mask_for_high_rank(val, val_mask, name=None):
    val_mask = tf.expand_dims(val_mask, -1)
    return tf.add(val, (1 - tf.cast(val_mask, tf.float32)) * VERY_NEGATIVE_NUMBER,
                  name=name or 'exp_mask_for_high_rank')

@tf.function
def mask_for_high_rank(val, val_mask, name=None):
    val_mask = tf.expand_dims(val_mask, -1)
    return tf.multiply(val, tf.cast(val_mask, tf.float32), name=name or 'mask_for_high_rank')


class Activation(keras.layers.Layer):
    def __init__(self, output_size, activation, **kwargs):
        super(Activation, self).__init__(**kwargs)
        self.output_size = output_size
        self.flatten = Flatten(1)
        self.leaner = keras.layers.Dense(output_size, activation=activation)
        self.reconstruct = Reconstruct(1)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.leaner(x)
        x = self.reconstruct([x, inputs])
        return x


class SumDence(keras.layers.Layer):
    def __init__(self, embedding_size, activation):
        super(SumDence, self).__init__()
        self.activation = Activation(embedding_size, activation)

    def call(self, inputs):
        inputs_embed, inputs_mask = inputs
        # tensor of [batch_size,n_visits,n_codes,embedding_size]
        valid_inputs_embed = mask_for_high_rank(inputs_embed, inputs_mask)
        # tensor of [batch_size,n_visits,vec]
        inputs_merged = tf.reduce_sum(valid_inputs_embed, 2)
        inputs_merged = self.activation(inputs_merged)
        return inputs_merged


class PrePostProcessingWrapper(tf.keras.layers.Layer):
  """Wrapper class that applies layer pre-processing and post-processing."""

  def __init__(self, layer, version):
    super(PrePostProcessingWrapper, self).__init__()
    self.layer = layer
    self.version = version
    self.layer_norm = LayerNormalization()


  def call(self, x):
    """Calls wrapped layer with same parameters."""

    x, mask = x
    if self.version == 'latest':
        # Preprocessing: apply layer normalization
        y = self.layer_norm(x)
        # Get layer output
        y = self.layer((y, mask))
        # Postprocessing: residual connection
        return x + y
    else:
        # Get layer output
        y = self.layer((x, mask))
        # Postprocessing: residual connection and layer normalization
        return self.layer_norm(x + y)


class EncoderStack(keras.layers.Layer):
    """Transformer encoder stack.

    The encoder stack is made up of N identical layers. Each layer is composed
    of the sublayers:
    1. Masked encoder layer
    2. Feedforward network (which is 2 fully-connected layers)
    """

    def __init__(self, params, train, version, embedding_size, **kwargs):
        super(EncoderStack, self).__init__(**kwargs)
        self.layers = []
        self.train = train
        self.version = version
        self.embedding_size = embedding_size
        # Create final layer normalization layer.
        self.output_normalization = LayerNormalization()
        for _ in range(params["num_hidden_layers"]):
            # Create sublayers for each layer.
            masked_encoder_layer = MultiHeadAttention(params["direction"],
                                                               train,
                                                               params["dropout"],
                                                               self.embedding_size,
                                                               params["num_heads"],
                                                               name='masked_encoder')
            feed_forward_network = FeedForwardNetwork(params["hidden_size"],
                                                                params["filter_size"],
                                                                params["dropout"],
                                                                train, params["allow_ffn_pad"],
                                                                name='feed_forward_network')

            self.layers.append([
                          PrePostProcessingWrapper(masked_encoder_layer,self.version),
                          PrePostProcessingWrapper(feed_forward_network,self.version)
                      ])


    def call(self, inputs):
        encoder_inputs, input_mask = inputs
        """Return the output of the encoder layer stacks.

        Args:
          encoder_inputs: tensor with shape [batch_size, number_visits, number_codes, hidden_size]
          input_mask: mask for the encoder self-attention layer.
            [batch_size, number_visits, number_codes]

        Returns:
          Output of encoder layer stack.
          float32 tensor with shape [batch_size, inumber_visits, number_codes, hidden_size]
        """
        for n, layer in enumerate(self.layers):
          # Run inputs through the sublayers.
          masked_encoder_layer = layer[0]
          feed_forward_network = layer[1]

          encoder_inputs = masked_encoder_layer((encoder_inputs, input_mask))
          encoder_inputs = feed_forward_network((encoder_inputs, input_mask))

        if self.version == 'latest':
            return self.output_normalization(encoder_inputs)
        else:
            return encoder_inputs