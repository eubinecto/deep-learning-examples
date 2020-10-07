from typing import List
import pprint

import tensorflow as tf
import tensorflow_datasets as tfds

# check out the documentation from here
# https://www.tensorflow.org/datasets/catalog/ted_hrlr_translate
TED_HRLR_PT_TO_EN_PATH = 'ted_hrlr_translate/pt_to_en'


def load_ted_hrlr():
    global TED_HRLR_PT_TO_EN_PATH
    examples, metadata = tfds.load(name=TED_HRLR_PT_TO_EN_PATH,
                                   # True: contains info associated with the builder
                                   with_info=True,
                                   # True: returns a tuple (input, label)
                                   # False: returns a dictionary (all the features)
                                   as_supervised=True)

    return examples, metadata


def main():
    pp = pprint.PrettyPrinter(indent=4)
    # https://www.tensorflow.org/datasets/overview#find_available_datasets
    print("### find available datasets ###")
    builders: List[str] = tfds.list_builders()
    pp.pprint(builders)
    examples, metadata = load_ted_hrlr()
    print("### examples ###")
    pp.pprint(examples)
    print("### metadata ###")
    pp.pprint(metadata)


if __name__ == "__main__":
    main()
