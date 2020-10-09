# Masking

## what do we need masking for?
1. Ignoring padded inputs (`padding_mask`) <- encoder, decoder
2. Self-attention causality(`look-ahead-mask`) <- decoder



## goals
- implementing the masking done in the decoder.
- Mask all the pad tokens in the batch of sequence.
- The look-ahead mask


## `create_padding_mask`

If you cast a boolean type to be float by using `tf.cast`, then
- True -> 1
- False -> 0

```python
import tensorflow as tf
bool_ex = tf.constant([True, False])
tf.cast(bool_ex, tf.float32)
Out[4]: <tf.Tensor: shape=(2,), dtype=float32, numpy=array([1., 0.], dtype=float32)>
```

## `create_look_ahead_mask`

```python
import tensorflow as tf
size = 5
ones_sq_mat = tf.ones((size, size))
ones_sq_mat
Out[57]: 
<tf.Tensor: shape=(5, 5), dtype=float32, numpy=
array([[1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.]], dtype=float32)>
tf.linalg.band_part(ones_sq_mat, 0, 0)
Out[58]: 
<tf.Tensor: shape=(5, 5), dtype=float32, numpy=
array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]], dtype=float32)>
tf.linalg.band_part(ones_sq_mat, 1, 0)
Out[59]: 
<tf.Tensor: shape=(5, 5), dtype=float32, numpy=
array([[1., 0., 0., 0., 0.],
       [1., 1., 0., 0., 0.],
       [0., 1., 1., 0., 0.],
       [0., 0., 1., 1., 0.],
       [0., 0., 0., 1., 1.]], dtype=float32)>
tf.linalg.band_part(ones_sq_mat, 2, 0)
Out[60]: 
<tf.Tensor: shape=(5, 5), dtype=float32, numpy=
array([[1., 0., 0., 0., 0.],
       [1., 1., 0., 0., 0.],
       [1., 1., 1., 0., 0.],
       [0., 1., 1., 1., 0.],
       [0., 0., 1., 1., 1.]], dtype=float32)>
tf.linalg.band_part(ones_sq_mat, 3, 0)
Out[61]: 
<tf.Tensor: shape=(5, 5), dtype=float32, numpy=
array([[1., 0., 0., 0., 0.],
       [1., 1., 0., 0., 0.],
       [1., 1., 1., 0., 0.],
       [1., 1., 1., 1., 0.],
       [0., 1., 1., 1., 1.]], dtype=float32)>
tf.linalg.band_part(ones_sq_mat, 4, 0)
Out[62]: 
<tf.Tensor: shape=(5, 5), dtype=float32, numpy=
array([[1., 0., 0., 0., 0.],
       [1., 1., 0., 0., 0.],
       [1., 1., 1., 0., 0.],
       [1., 1., 1., 1., 0.],
       [1., 1., 1., 1., 1.]], dtype=float32)>
tf.linalg.band_part(ones_sq_mat, -1, 0)
Out[63]: 
<tf.Tensor: shape=(5, 5), dtype=float32, numpy=
array([[1., 0., 0., 0., 0.],
       [1., 1., 0., 0., 0.],
       [1., 1., 1., 0., 0.],
       [1., 1., 1., 1., 0.],
       [1., 1., 1., 1., 1.]], dtype=float32)>
look_ahead_mask_sq_mat = 1 - tf.linalg.band_part(ones_sq_mat, -1, 0)
look_ahead_mask_sq_mat
Out[65]: 
<tf.Tensor: shape=(5, 5), dtype=float32, numpy=
array([[0., 1., 1., 1., 1.],
       [0., 0., 1., 1., 1.],
       [0., 0., 0., 1., 1.],
       [0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0.]], dtype=float32)>

```

the goal is to.. 

## questions

> Mask all the pad tokens in the batch of sequence. 
>It ensures that the model does not treat padding as the input.
> The mask indicates where pad value 0 is present: it outputs a 1 at those locations,
> and a 0 otherwise.

### but how does it "ensure that it is not an input"? 

scaled-dot-attention block|
--- |
![](https://www.tensorflow.org/images/tutorials/transformer/scaled_attention.png) |

```python
 # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9) 
```
> The mask is multiplied with -1e9 (close to negative infinity).
> This is done because the mask is summed with the scaled matrix multiplication of
> Q and K and is applied immediately before a softmax. 
>The goal is to **zero out these cells**, 
>and large negative inputs to softmax are near zero in the output.

그냥 0으로 냅두면 attention logit이 0에 수렴하지 않음.


### why do you add extra dimension here?
`scaled_attention_logits`의 dimension이 그렇기 때문에? 이 부분은 더 이해가 필요하다.


## references
- [the two purposes of masking in transformer](https://datascience.stackexchange.com/a/65070)
- [illustrated transformer](http://jalammar.github.io/illustrated-transformer/)
