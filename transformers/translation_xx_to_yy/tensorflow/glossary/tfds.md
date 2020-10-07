# tfds
## What?

> a collection of ready-to-use datasets.

2019년 상반기 TF Dev submit에서 처음 소개된 데이터 로드 라이브러리.


## Why?

slide|설명
---|---|
![](.README_images/70328216.png)| `tf.data`를 활용해서 유연한 파이프라인이 가능했지만
![](.README_images/b76d69fc.png)| 데이터 자체에는 통일성이 없다. MNIST, IMDB - 다 조금씩 다르다.

새로운 데이터가 published 될 때 마다, 서로 조금씩 다른 포맷으로 나온다. 때문에 `tf.data` 파이프라인으로 데이터를 
조작하기 위해선 데이터 셋 별로 전처리를 매번 해주어야 한다.

데이터를 publish하는 포맷을 `tf.data` 파이프라인을 연결하기 쉽도록 하는 하나의 포맷으로 통일해버리면 어떨까? -> tensorflow_datasets.

요약을 하자면:
> Datasets are distributed in all kinds of formats and in all kinds of places, 
>and they're not always stored in a format that's ready to feed into a machine learning pipeline. Enter TFDS.



## How?
### 사용가능한 데이터 셋 확인
```python
pp = pprint.PrettyPrinter(indent=4)
    # https://www.tensorflow.org/datasets/overview#find_available_datasets
    print("### find available datasets ###")
    builders: List[str] = tfds.list_builders()
    pp.pprint(builders)
```


```text
[   'abstract_reasoning',
    'aeslc',
    'aflw2k3d',
    'ag_news_subset',
    'ai2_arc',
    'ai2_arc_with_ir',
    'amazon_us_reviews',
    'anli',
    'arc',
    'bair_robot_pushing_small',
    'bccd',
    'beans,
     ...,
]
```

### 원하는 데이터 셋 로드

slide | 설명
--- | ---|
![](.README_images/25458e81.png) | `tfds.load`로 데이터를 지정하고, 어떤 포맷으로 나오길 바라는지 지정할 수 있다. 

[`tfds.load` API](https://www.tensorflow.org/datasets/api_docs/python/tfds/load) 문서 참고하기. 정말 잘 쓰여져 있음
### 직접 만든 데이터 셋을 업로드

[공식 Instructions](https://www.tensorflow.org/datasets/add_dataset) 에 따라 직접 자신의 데이터 셋을
전처리해서 distribute 할 수 있다.