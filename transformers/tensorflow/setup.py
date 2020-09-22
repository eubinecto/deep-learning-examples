
import tensorflow as tf
import tensorflow_datasets as tfds

# check out the documentation from here
# https://www.tensorflow.org/datasets/catalog/ted_hrlr_translate
TED_HRLR_PT_TO_EN_PATH = 'ted_hrlr_translate/pt_to_en'


def load_datasets():
    global TED_HRLR_PT_TO_EN_PATH
    example, metadata = tfds.load(name=TED_HRLR_PT_TO_EN_PATH,
                                  # True: contains info associated with the builder
                                  with_info=True,
                                  # True: returns a tuple (input, label)
                                  # False: returns a dictionary (all the features)
                                  as_supervised=True)
    pass


def main():
    load_datasets()


if __name__ == "__main__":
    main()