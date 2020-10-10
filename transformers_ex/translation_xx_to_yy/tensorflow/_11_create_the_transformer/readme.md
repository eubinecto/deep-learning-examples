
# Transformer


bird-eyes' view of the transformer|
---|
![](http://jalammar.github.io/images/t/transformer_resideual_layer_norm_3.png)|



## the output
```
---transformer out:
<tf.Tensor: shape=(64, 36, 8000), dtype=float32, numpy=
```
- 64 sentences
- all of which are of length 36
- the output is logit distribution across all the words in the vocab, for each time stamp.

