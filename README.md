## Install Requrements

    pip install -r requrements.txt

## Data

The dataset used in this project is **Stanford Cars Dataset** [[1]](#1), you can download it [here](http://ai.stanford.edu/~jkrause/cars/car_dataset.html).

### Default Directory Path

```
├── cars_train
│   ├── 00001.jpg
│   ├── 00002.jpg
│   ├── 00003.jpg
│   ├── ...
│   └── 08144.jpg
├── cars_test
│   ├── 00001.jpg
│   ├── 00002.jpg
│   ├── 00003.jpg
│   ├── ...
│   └── 08041.jpg
├── cars_meta.mat
├── cars_train_annos.mat
└── cars_test_annos_withlabels.mat
```

## Available Model

You can see the complete models & weights specification [here](https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights).

| Model Name | Pre-trained | Pre-trained Acc@1 | Pre-trained Acc@5 | Params |
| ---------- | :----------: | :----------------: | :----------------: | :------: |
| 'mobilenet_v3_l' | imagenet | 75.274 | 92.566 | 5.5M |
| 'efficientnet_v2_s' | imagenet | 84.228 | 96.878 | 21.5M |

## Training

### Training Example

```console
python train.py --data data --output output --model 'mobilenet_v3_l' --epoch 50 --image_size 240 360 --batch_size 32
```

### Resume Training Example

```console
python train.py --data data --output output --model 'mobilenet_v3_l' --epoch 50 --image_size 240 360 --batch_size 32 --checkpoint output/checkpoint.pt
```

### Available Argument for Training

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| --data | str | 'data' | Path of your dataset location |
| --output | str | 'output' | Path of your output location |
| --model | str | 'mobilenet_v3_l' | Model that will be used for training |
| --epoch | int | 50 | How many epoch your model will be trained |
| --image_size | int | 240 360 | Image size (h x w) |
| --batch_size | int | 32 | Batch size for training |
| --learning_rate | float | 0.001 | Learning rate for training |
| --momentum | float | 0.9 | Momentum used for training |
| --checkpoint | str | None | Path of your checkpoint file for resuming the training process|
| --device | str | 'cuda' | Device used for training, either cuda (gpu) or cpu |

## Testing

### Testing Example

```console
python test.py --data data --model 'mobilenet_v3_l' --weights output/weights.pt
```

### Available Argument for Testing

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| --data | str | 'data' | Path of your dataset location |
| --model | str | 'mobilenet_v3_l' | Model that will be used for testing |
| --weights | str | None | Path of your trained model weights location |
| --image_size | int | 240 360 | Image size (h x w) |
| --device | str | 'cuda' | Device used for testing, either cuda (gpu) or cpu |

## Inference

### Inference Example

```console
python inference.py --data data --model 'mobilenet_v3_l' --weights output/weights.pt --image data\cars_test\00001.jpg
```

### Available Argument for Inference

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| --data | str | 'data' | Path of your dataset location |
| --model | str | 'mobilenet_v3_l' | Model that will be used for inference |
| --weights | str | None | Path of your trained model weights location |
| --image | str | None | Path of the image that you want to inference |
| --image_size | int | 240 360 | Image size (h x w) |
| --device | str | 'cuda' | Device used for inference, either cuda (gpu) or cpu |

## Reference
<a id='1'>[1]</a> 
J. Krause, M. Stark, J. Deng, and L. Fei-Fei,
'3D Object Representations for Fine-Grained Categorization', 
4th IEEE Workshop on 3D Representation and Recognition, in 4th International IEEE Workshop on 3D Representation and Recognition (3dRR-13), 2013.