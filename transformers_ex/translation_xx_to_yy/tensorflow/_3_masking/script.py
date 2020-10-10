import tensorflow as tf
import pprint


class Masking:
    @classmethod
    def create_padding_mask(cls, seq: tf.Tensor) -> tf.Tensor:
        # tf.math.equal: if encoded val is 0 -> True else -> False.
        # tf.cast(bool, tf.float) ->
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    @classmethod
    def create_look_ahead_mask(cls, size: int) -> tf.Tensor:
        """
        The look-ahead mask is used to mask the future tokens in a sequence.
        In other words, the mask indicates which entries should not be used.
        This means that to predict the third word, only the first and second
        word will be used. Similarly to predict the fourth word,
        only the first, second and the third word will be used and so on.
        """
        sq_mat_ones = tf.ones((size, size))
        mask = 1 - tf.linalg.band_part(input=sq_mat_ones,
                                       # reserve all the lower diagonals
                                       num_lower=-1,
                                       # reserve
                                       num_upper=0)
        return mask  # (seq_len, seq_len)


def main():
    pp = pprint.PrettyPrinter(indent=4)
    print("### Testing the padding mask ####")
    x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
    x_masked = Masking.create_padding_mask(x)
    pp.pprint(x)
    print("--- padding mask for this is: -----")
    pp.pprint(x_masked)

    print("\n### Testing the look-ahead mask ###")
    # outputs random values from a uniform distribution, with the given shape
    y = tf.random.uniform((1, 3))
    y_masked = Masking.create_look_ahead_mask(size=y.shape[1])  # the size is the length of the seq
    pp.pprint(y)
    print("--- look ahead mask for this is: -----")
    pp.pprint(y_masked)


if __name__ == '__main__':
    main()
