# encoder

stack encoder layers to complete the encoder|
--- |
![](http://jalammar.github.io/images/t/The_transformer_encoder_decoder_stack.png)|


## question
why multiply square root?
```python
x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # < -- why?

```