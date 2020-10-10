from transformers_ex.translation_xx_to_yy.tensorflow._1_setup_input_pipeline.script import SetupInputPipeline

NUM_LAYERS = 4
D_MODEL = 128
DFF = 512
NUM_HEADS = 8

pipeline = SetupInputPipeline()
pipeline.init_subwords_tokenizer()
INPUT_VOCAB_SIZE = pipeline.tokenizer_pt.vocab_size + 2
TARGET_VOCAB_SIZE = pipeline.tokenizer_en.vocab_size + 2
DROPOUT_RATE = 0.1
