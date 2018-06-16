# PixelVAE

Code for the models in [PixelVAE: A Latent Variable Model for Natural Images](https://arxiv.org/abs/1611.05013).

# Files by Leila Islam...

## Real-valued MNIST Custom Sample Generation

To generate samples for a pre-trained model whose weights are already stored in the same folder:

```
CUDA_VISIBLE_DEVICES=0,1 python pixelvaesamples.py
```

By default, this runs on real-valued MNIST. You can specify how many images "num" to generate by going into the file.

To encode one or more images and then generate samples for a pre-trained distribution whose weights are stored in the same folder:

```
CUDA_VISIBLE_DEVICES=0,1 python pixelvaeinterpolator.py
```

By default, this runs on real-valued MNIST. This is currently being debugged.

# Files by original authors....

## Binarized MNIST

To train:

```
python models/mnist_pixelvae_train.py -L 12 -fs 5 -algo cond_z_bias -dpx 16 -ldim 16
```

To evaluate, take the weights of the model with best validation score from the above training procedure and then run

```
python models/mnist_pixelvae_evaluate.py -L 12 -fs 5 -algo cond_z_bias -dpx 16 -ldim 16 -w path/to/weights.pkl
```

## Real-valued MNIST, LSUN Bedrooms, 64x64 ImageNet

To train, evaluate, and generate samples:

```
python pixelvaecheckpoints.py
```

By default, this runs on real-valued MNIST. You can specify different datasets or model settings within the file.

