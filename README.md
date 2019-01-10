# keras_layerNorm_mlp_lstm
layer normalization for mlp and lstm in Keras

ln_lstm reference from https://github.com/cleemesser/keras-layer-norm-work

ln_mlp reference from https://github.com/CyberZHG/keras-layer-normalization

download imdb data from http://s3.amazonaws.com/text-datasets/aclImdb.zip

download pretrained glove file from https://nlp.stanford.edu/projects/glove

## Layer Norm for MLP

By comparing vanilla, BN and LN in full connected part of CNN on MNIST dataset

```py
model_ln = Sequential()
model_ln.add(Conv2D(input_shape = (28,28,1), filters=6, kernel_size=(5,5), padding='valid', activation='tanh'))
model_ln.add(MaxPool2D(pool_size=(2,2), strides=2))
model_ln.add(Conv2D(input_shape=(14,14,6), filters=16, kernel_size=(5,5), padding='valid', activation='tanh'))
model_ln.add(MaxPool2D(pool_size=(2,2), strides=2))
model_ln.add(Flatten())
model_ln.add(Dense(120, activation='tanh'))
model_ln.add(LayerNormalization()) # 添加LN运算
model_ln.add(Dense(84, activation='tanh'))
model_ln.add(LayerNormalization())
model_ln.add(Dense(10, activation='softmax'))
```

batch_size = 128 (BN > LN >> Vanilla)

![](/image/mnist_128.jpg)

batch_size = 8 (LN > vanilla >> BN)

![](/image/mnist_8.jpg)

## Layer Norm for LSTM

By comparing vanilla and LN in LSTM on imdb dataset (LN > vanilla)

```py
# https://github.com/cleemesser/keras-layer-norm-work
from lstm_ln import LSTM_LN
model_ln = Sequential()

model_ln.add(Embedding(max_features,100))
model_ln.add(LSTM_LN(128))
model_ln.add(Dense(1, activation='sigmoid'))
model_ln.summary()
```
Training loss

![](/image/imdb_loss.jpg)

Val accuracy

![](/image/imdb_val_acc.jpg)
