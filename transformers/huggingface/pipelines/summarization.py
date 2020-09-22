from transformers import pipeline, SummarizationPipeline, Pipeline
from typing import cast
import logging
logging.basicConfig(level=logging.INFO)

# pipeline for summarising sentences. you can define the min & max length of the summary.


TEST_SENTENCES = [

    # short one
    "Sam Shleifer writes the best docstring examples in the whole world.",

    # lone one - (the abstract of attention is all you need paper)
    """The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks that include an encoder and a decoder. The best
performing models also connect the encoder and decoder through an attention
mechanism. We propose a new simple network architecture, the Transformer,
based solely on attention mechanisms, dispensing with recurrence and convolutions
entirely. Experiments on two machine translation tasks show these models to
be superior in quality while being more parallelizable and requiring significantly
less time to train. Our model achieves 28.4 BLEU on the WMT 2014 Englishto-
German translation task, improving over the existing best results, including
ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task,
our model establishes a new single-model state-of-the-art BLEU score of 41.8 after
training for 3.5 days on eight GPUs, a small fraction of the training costs of the
best models from the literature. We show that the Transformer generalizes well to
other tasks by applying it successfully to English constituency parsing both with
large and limited training data."""

]


TASK_ID = "summarization"


def main():
    logger = logging.getLogger("main")
    # use bert in pytorch (this is the default)
    # The model will take up about 1.5G
    logger.info("loading pipeline...")
    summarizer: cast(SummarizationPipeline, Pipeline) = pipeline("summarization")
    results = summarizer(TEST_SENTENCES, min_length=5, max_length=20)

    # report
    for sent, res in zip(TEST_SENTENCES, results):
        print("---")
        print("IN:\n" + sent)
        print("OUT:\n" + res["summary_text"])


if __name__ == "__main__":
    main()
