from typing import Tuple
import pprint
import tensorflow as tf
# using the scaled_dot_product_attention
from transformers_ex.translation_xx_to_yy.tensorflow._4_scaled_dot_product_attention.script import scaled_dot_product_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model: int, num_heads: int):
        """
        d_model: dimension of the model. (512)
        num_heads: self-explanatory
        """

        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        # because each head is split to attend to different
        # parts of the dimensionality.
        # e.g. d_model=512, num_heads = 4 -> ([:127], [127:255], ... )
        assert d_model % self.num_heads == 0

        # what is depth here?
        self.depth = d_model // self.num_heads

        # attach weights to train
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        #
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size) -> tf.Tensor:
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask) -> Tuple[tf.Tensor, tf.Tensor]:
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def main():
    pp = pprint.PrettyPrinter(indent=4)
    print("### Testing Multi-head attention ###")
    temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
    # a sentence of length 60, where the dimensionality of the embedding vector is 512.
    # (batch_size, encoder_sequence, d_model)
    y = tf.random.uniform((1, 60, 512))
    print("--- input:")
    pp.pprint(y)
    print("--- the shape of out:")
    out, attn = temp_mha(v=y, k=y, q=y, mask=None)
    pp.pprint(out.shape)
    print("--- the shape of attn:")
    pp.pprint(attn.shape)



if __name__ == '__main__':
    main()
