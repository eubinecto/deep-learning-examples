# training and checkpointing

## script out
```
/Users/eubin/Desktop/Books/deep-learning-examples/dlegpyenv/bin/python /Users/eubin/Desktop/Books/deep-learning-examples/transformers_ex/translation_xx_to_yy/tensorflow/_15_training_and_checkpointing/script.py
INFO:create_subwords_tokenizer:tokenizer_pt has been loaded from cache
INFO:create_subwords_tokenizer:tokenizer_en has been loaded from cache
INFO:create_subwords_tokenizer:tokenizer_pt has been loaded from cache
INFO:create_subwords_tokenizer:tokenizer_en has been loaded from cache
2020-10-10 09:55:12.789798: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2020-10-10 09:55:12.800419: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7fc96b68c140 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-10-10 09:55:12.800430: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
INFO:absl:Load dataset info from /Users/eubin/tensorflow_datasets/ted_hrlr_translate/pt_to_en/1.0.0
INFO:absl:Reusing dataset ted_hrlr_translate (/Users/eubin/tensorflow_datasets/ted_hrlr_translate/pt_to_en/1.0.0)
INFO:absl:Constructing tf.data.Dataset for split None, from /Users/eubin/tensorflow_datasets/ted_hrlr_translate/pt_to_en/1.0.0
INFO:create_subwords_tokenizer:tokenizer_pt has been loaded from cache
INFO:create_subwords_tokenizer:tokenizer_en has been loaded from cache
Epoch 1 Batch 0 Loss 9.0361 Accuracy 0.0000
Epoch 1 Batch 50 Loss 8.9834 Accuracy 0.0004
```