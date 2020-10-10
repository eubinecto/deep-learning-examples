from typing import List, Tuple, Optional
from os import path
import pickle
import pprint
import tensorflow as tf
import tensorflow_datasets as tfds
import itertools
import logging
import sys
logging.basicConfig(stream=sys.stdout,  # so that colors are not all red.
                    level=logging.INFO)

# maybe, try loading dataset from cache? you could use pickle library for this.


class SetupInputPipeline:
    """
    This class is written based off this
    https://www.tensorflow.org/datasets/catalog/ted_hrlr_translate
    """
    # directories
    MODULE_DIR = "/".join(__file__.split("/")[:-1])
    CACHE_DIR = path.join(MODULE_DIR, ".cache_pkl")
    TOKENIZER_PT_PATH = path.join(CACHE_DIR, "tokenizer_pt.pkl")
    TOKENIZER_EN_PATH = path.join(CACHE_DIR, "tokenizer_en.pkl")
    TED_HRLR_PT_TO_EN_PATH = 'ted_hrlr_translate/pt_to_en'
    # the size of the entire vocab.
    APPROX_VOCAB_SIZE = 2**13
    # For perfect shuffling, a buffer size greater than or equal to the full size of the dataset is required.
    BUFFER_SIZE = 20000
    BATCH_SIZE = 64  # < why is the size of the batch so small here?
    MAX_LENGTH = 40  # to keep the training relatively fast.
    # these datasets are to be loaded
    train_ex: Optional[tf.data.Dataset] = None
    val_ex: Optional[tf.data.Dataset] = None
    test_ex: Optional[tf.data.Dataset] = None
    metadata: Optional[tfds.core.DatasetInfo] = None
    # these tokenizer is to be filled
    tokenizer_pt: Optional[tfds.deprecated.text.SubwordTextEncoder] = None
    tokenizer_en: Optional[tfds.deprecated.text.SubwordTextEncoder] = None

    def init_ted_hrlr(self):
        """
        loads ted hrlr dataset from tfds.
        if the data does not exist in ~/tensorflow_datasets, then tfds.load
        will start downloading the dataset from the web.
        tfds.load: https://www.tensorflow.org/datasets/api_docs/python/tfds/load
        """
        data: Tuple[tf.data.Dataset, tfds.core.DatasetInfo] = \
            tfds.load(name=self.TED_HRLR_PT_TO_EN_PATH,
                      # True: contains info associated with the builder
                      with_info=True,
                      # True: returns the examples with tuple structure (input, label)
                      as_supervised=True)
        examples = data[0]
        metadata = data[1]  # we get this because with_info was set to True.
        # take out the train
        # since as_supervised is set to true, these will have 2-tuple structure (input, label)
        self.train_ex = examples['train']
        self.val_ex = examples['validation']
        self.test_ex = examples['test']
        self.metadata = metadata

    def init_subwords_tokenizer(self):
        """
        given a ted dataset (pt sent -> eng sent), returns the tokenizers for each lang.
        tokenizer breaks a sentence down to words.
        A better approach than this: use an embedding layer.
        SubwordTextEncoder: https://www.tensorflow.org/datasets/api_docs/python/tfds/deprecated/text/SubwordTextEncoder
        """
        logger = logging.getLogger("create_subwords_tokenizer")
        if path.exists(self.TOKENIZER_PT_PATH) \
                and path.exists(self.TOKENIZER_EN_PATH):
            # load the pt tokenizer from cache
            with open(self.TOKENIZER_PT_PATH, 'rb') as fh:
                self.tokenizer_pt = pickle.loads(fh.read())
                logger.info("tokenizer_pt has been loaded from cache")
            # load the en tokenizer from cache
            with open(self.TOKENIZER_EN_PATH, 'rb') as fh:
                self.tokenizer_en = pickle.loads(fh.read())
                logger.info("tokenizer_en has been loaded from cache")
        else:
            # build the tokenizers from the training set
            gen_pt_sent = (pt.numpy() for (pt, _) in self.train_ex)
            gen_en_sent = (en.numpy() for (_, en) in self.train_ex)
            logger.info("building tokenizer for pt...")
            tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
                # any generator that outputs a string
                corpus_generator=gen_pt_sent,
                # approximate size of the vocabulary to create
                target_vocab_size=self.APPROX_VOCAB_SIZE
            )
            logger.info("building tokenizer for en...")
            tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
                corpus_generator=gen_en_sent,
                target_vocab_size=self.APPROX_VOCAB_SIZE
            )
            self.tokenizer_pt = tokenizer_pt
            self.tokenizer_en = tokenizer_en
            # cache the tokenizer for later use
            self.cache_subwords_tokenizer()

    def cache_subwords_tokenizer(self):
        logger = logging.getLogger("cache_subwords_tokenizer")
        assert self.tokenizer_pt and self.tokenizer_en
        # cache the pt tokenizer
        with open(self.TOKENIZER_PT_PATH, 'wb') as fh:
            fh.write(pickle.dumps(self.tokenizer_pt))
            logger.info("tokenizer_pt has been cached.")
        with open(self.TOKENIZER_EN_PATH, 'wb') as fh:
            fh.write(pickle.dumps(self.tokenizer_en))
            logger.info("tokenizer_en has been cached.")

    def encode_start_end_token(self,
                               pt_sent: tf.Tensor,
                               en_sent: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        mapper function to encode start-of-seq and end-of-seq tokens to each datapoint
        in the dataset.
        return: returns two tensors as a tuple
        """
        # assert that tokenizers are initialised
        assert self.tokenizer_en and self.tokenizer_pt
        # utilise the vocabulary size for the SOS and EOS tokens
        # this makes sense because the value of tokens ranges from 0 to (vocab_size - 1)
        PT_SOS = self.tokenizer_pt.vocab_size
        PT_EOS = self.tokenizer_pt.vocab_size + 1
        EN_SOS = self.tokenizer_en.vocab_size
        EN_EOS = self.tokenizer_en.vocab_size + 1
        # list concatenation - attach SOS up front and EOS at the end
        enc_pt_sent = [PT_SOS] + self.tokenizer_pt.encode(pt_sent.numpy()) + [PT_EOS]
        enc_en_sent = [EN_SOS] + self.tokenizer_en.encode(en_sent.numpy()) + [EN_EOS]
        return enc_pt_sent, enc_en_sent

    def tf_encode_start_end_token(self,
                                  pt_sent: tf.Tensor,
                                  en_sent: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        A wrapper for encode_start_end_token.
        To be used with Dataset.mapper.
        tf.py_function: https://www.tensorflow.org/api_docs/python/tf/py_function
        returns: tuples of two tensors, containing integers
        """
        # to apply encoding transformation to each instance in the training set,
        # we can use tf.py_function.
        enc_pt_sent, enc_en_sent = tf.py_function(func=self.encode_start_end_token,
                                                  inp=[pt_sent, en_sent],
                                                  # returns two tensors as a tuple
                                                  Tout=[tf.int64, tf.int64])
        enc_pt_sent.set_shape([None])  # < - but why do we need this?
        enc_en_sent.set_shape([None])
        return enc_pt_sent, enc_en_sent

    def filter_max_length(self,
                          pt_sent: tf.Tensor,
                          en_sent: tf.Tensor):
        """
        To keep this example small and relatively fast, drop examples with a length of over 40 tokens.
        To be used with dataset.filter.
        """
        return tf.logical_and(tf.size(pt_sent) <= self.MAX_LENGTH,
                              tf.size(en_sent) <= self.MAX_LENGTH)

    def preproc_train_ex(self) -> tf.data.Dataset:
        # assert that train example is initialised
        assert self.train_ex
        # filter: tensors with length <= 40
        # map: encode with the defined tokenizers, insert SOS and EOS.
        # cache: save elements that have been iterated
        # prefetch:  This allows later elements to be prepared while the current element is being processed.
        # This often improves latency and throughput,
        # at the cost of using additional memory to store prefetched elements.
        return self.train_ex.filter(predicate=self.filter_max_length)\
                            .map(map_func=self.tf_encode_start_end_token)\
                            .cache()\
                            .shuffle(buffer_size=self.BUFFER_SIZE)\
                            .padded_batch(batch_size=self.BATCH_SIZE)\
                            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


def main():
    pp = pprint.PrettyPrinter(indent=4)

    print("### looking up available datasets ###")
    builders: List[str] = tfds.list_builders()
    pp.pprint(builders)

    print("\n### the first 5 elements of train_ex ###")
    pipeline = SetupInputPipeline()
    pipeline.init_ted_hrlr()
    train_ex = pipeline.train_ex
    train_ex_numpy = ((pt.numpy(), en.numpy()) for (pt, en) in train_ex)  # generator comprehension
    pp.pprint(list(itertools.islice(train_ex_numpy, 5)))

    print("\n### testing the english tokenizer ###")
    pipeline.init_subwords_tokenizer()
    tokenizer_en = pipeline.tokenizer_en
    sample_string = "Transformer is awesome"
    tokenized_string = tokenizer_en.encode(s=sample_string)  # this won't terminate?
    print("Sample string: {}\nEncoded string: {}".format(sample_string, tokenized_string))

    print("\n### if a word does not exist in vocab, it is tokenized into subwords ###")
    for ts in tokenized_string:
        print('{} ----> {}'.format(ts, tokenizer_en.decode([ts])))

    print("\n### the first 5 elements of the preprocessed training set ###")
    preproc_train_ex = pipeline.preproc_train_ex()
    preproc_train_ex_numpy = ((pt.numpy(), en.numpy()) for (pt, en) in preproc_train_ex)
    # have a look at the first batch (64 instances)
    pp.pprint(next(iter(preproc_train_ex_numpy)))


if __name__ == "__main__":
    main()
