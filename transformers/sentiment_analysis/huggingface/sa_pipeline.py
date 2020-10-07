from transformers import pipeline, TextClassificationPipeline, Pipeline
from typing import cast
import logging
logging.basicConfig(level=logging.INFO)

# documentation
# https://huggingface.co/transformers/main_classes/pipelines.html#transformers.TextClassificationPipeline

# sentences to run sentiment analysis
TEST_SENTENCES = [
    "I hate you.",  # obviously negative
    "I love you.",  # obviously positive
    "You have a high blood pressure.",
    "Don't beat around the bush!",  # not obviously negative
    "I sighed in relief to know that the intellectual future of humanity is safe."  # not obviously positive
]

# his text classification pipeline can currently be loaded from pipeline() using the following task identifier
TASK_ID = 'sentiment-analysis'


def main():
    # hugging face adopts builder pattern for building pipelines.
    # the default model is bert.
    # if the model has not been downloaded, it will download the model on the fly
    logger = logging.getLogger("main")
    logger.info("loading text classification pipeline...")
    bert_pipeline: cast(TextClassificationPipeline, Pipeline) = pipeline(TASK_ID, return_all_scores=True)
    # this will execute the __call__ implementation of TextClassificationPipeline
    # accepts a string, or list of strings as the parameter
    results = bert_pipeline(TEST_SENTENCES)

    # check out the results
    for sen, classes in zip(TEST_SENTENCES, results):
        print("---")
        # 0: negative, 1: positive
        classes: list
        print(sen)
        print(classes[0]['label'], ":", classes[0]['score'])
        print(classes[1]['label'], ":", classes[1]['score'])


if __name__ == "__main__":
    main()
