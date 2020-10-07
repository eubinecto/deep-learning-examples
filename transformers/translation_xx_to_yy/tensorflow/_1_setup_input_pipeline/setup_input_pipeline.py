from typing import List, Tuple
import pprint
import tensorflow as tf
import tensorflow_datasets as tfds
import itertools

# check out the documentation from here
# https://www.tensorflow.org/datasets/catalog/ted_hrlr_translate
TED_HRLR_PT_TO_EN_PATH = 'ted_hrlr_translate/pt_to_en'
APPROX_VOCAB_SIZE = 2**13
BUFFER_SIZE = 20000  # <- what does this number mean here?
BATCH_SIZE = 64  # < why is the size of the batch so small here?
MAX_LENGTH = 40  # to keep the training relatively fast.


def load_ted_hrlr() -> Tuple[tf.raw_ops.PrefetchDataset,
                             tf.raw_ops.PrefetchDataset,
                             tf.raw_ops.PrefetchDataset,
                             tfds.core.DatasetInfo]:
    """
    loads ted hrlr dataset from tfds.
    if the data does not exist in ~/tensorflow_datasets, then tfds.load
    will start downloading the dataset from the web.
    tfds.load: https://www.tensorflow.org/datasets/api_docs/python/tfds/load
    """
    global TED_HRLR_PT_TO_EN_PATH
    data: Tuple[tf.data.Dataset, tfds.core.DatasetInfo] = \
        tfds.load(name=TED_HRLR_PT_TO_EN_PATH,
                  # True: contains info associated with the builder
                  with_info=True,
                  # True: returns the examples with tuple structure (input, label)
                  as_supervised=True)
    examples = data[0]
    metadata = data[1]  # we get this because with_info was set to True.
    # take out the train
    # since as_supervised is set to true, these will have 2-tuple structure (input, label)
    train_ex: tf.raw_ops.PrefetchDataset = examples['train']
    val_ex: tf.raw_ops.PrefetchDataset = examples['validation']
    test_ex: tf.raw_ops.PrefetchDataset = examples['test']
    return train_ex, val_ex, test_ex, metadata


def create_subwords_tokenizer(ted_ex: tf.data.Dataset) -> \
        Tuple[tfds.deprecated.text.SubwordTextEncoder,
              tfds.deprecated.text.SubwordTextEncoder]:
    """
    given a ted dataset (pt sent -> eng sent), returns the tokenizers for each lang.
    tokenizer breaks a sentence down to words.
    SubwordTextEncoder: https://www.tensorflow.org/datasets/api_docs/python/tfds/deprecated/text/SubwordTextEncoder
    """
    global APPROX_VOCAB_SIZE
    gen_pt_sent = (pt.numpy() for (pt, _) in ted_ex)
    gen_en_sent = (en.numpy() for (_, en) in ted_ex)
    tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        # any generator that outputs a string
        corpus_generator=gen_pt_sent,
        # approximate size of the vocabulary to create
        target_vocab_size=APPROX_VOCAB_SIZE
    )
    tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        corpus_generator=gen_en_sent,
        target_vocab_size=APPROX_VOCAB_SIZE
    )
    # return the two tokenizers
    return tokenizer_pt, tokenizer_en


def main():
    pp = pprint.PrettyPrinter(indent=4)
    print("### looking up available datasets ###")
    builders: List[str] = tfds.list_builders()
    pp.pprint(builders)
    train_ex, val_ex, test_ex, metadata = load_ted_hrlr()
    print("### metadata ###")
    pp.pprint(metadata)  # you can check the supervised_keys here
    print("### the first 10 elements of train_ex ###")
    train_ex_numpy = ((pt.numpy(), en.numpy()) for (pt, en) in train_ex)  # generator comprehension
    pp.pprint(list(itertools.islice(train_ex_numpy, 10)))
    print("### testing the english tokenizer ###")
    tokenizer_en, _ = create_subwords_tokenizer(ted_ex=train_ex)
    sample_string = "Transformer is awesome"
    encoded_string = tokenizer_en.encode(s=sample_string)  # this won't terminate?
    pp.pprint(encoded_string)


if __name__ == "__main__":
    main()
