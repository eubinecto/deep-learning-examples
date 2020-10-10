from transformers_ex.translation_xx_to_yy.tensorflow._2_positional_encoding.script import positional_encoding
from transformers_ex.translation_xx_to_yy.tensorflow._7_encoder_layer.script import EncoderLayer
import tensorflow as tf
import pprint


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        # embedding layer is jointly trained
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        # the encoder stack
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # < -- why?
        # select all batch, up to the sequence length, all dimension
        x += self.pos_encoding[:, :seq_len, :]
        # to increase the performance
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            # the out of the prev layer to the in of the next layer
            x = self.enc_layers[i](x, training, mask)
        return x  # (batch_size, input_seq_len, d_model)


if __name__ == '__main__':
    print("### Testing Encoder ###")
    pp = pprint.PrettyPrinter(indent=4)
    sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8,
                             dff=2048, input_vocab_size=8500,
                             maximum_position_encoding=10000)
    temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)
    print("---input is:")
    pp.pprint(temp_input)
    print("---output is:")
    sample_encoder_output = sample_encoder(temp_input, training=False, mask=None)
    print(sample_encoder_output)  # (batch_size, input_seq_len, d_model)
    print("---the shape of the output is:")
    pp.pprint(sample_encoder_output.shape)
