# Scaled Dot Product Attention






## script output
```
### Testing Multi-head attention ###
2020-10-10 08:17:58.848732: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2020-10-10 08:17:58.859600: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7f9880f7faf0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-10-10 08:17:58.859611: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
--- input:
<tf.Tensor: shape=(1, 60, 512), dtype=float32, numpy=
array([[[0.51487017, 0.58742356, 0.6084275 , ..., 0.38652682,
         0.4547143 , 0.3821218 ],
        [0.95432234, 0.46414685, 0.20054173, ..., 0.27764225,
         0.7018851 , 0.6230829 ],
        [0.3118341 , 0.9738172 , 0.50318015, ..., 0.62296534,
         0.8079902 , 0.51904607],
        ...,
        [0.18102634, 0.73430204, 0.438591  , ..., 0.8505206 ,
         0.5842401 , 0.12552035],
        [0.6678592 , 0.37700284, 0.19274831, ..., 0.3465526 ,
         0.37620962, 0.14977133],
        [0.54141927, 0.2282598 , 0.5224166 , ..., 0.08114564,
         0.6759881 , 0.7883817 ]]], dtype=float32)>
--- the shape of out:
TensorShape([1, 60, 512])
--- the shape of attn:
TensorShape([1, 8, 60, 60])

```


## Questions

### How do I subclass `tf.keras.layer.Layer` to build my own layer?



