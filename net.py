"Module where we define the network."

from keras.models import Model
from keras.layers import Input, Concatenate, BatchNormalization, Conv2D, MaxPooling2D, Deconv2D
from keras.activations import relu, sigmoid

def conv_relu(filters):
    """Conv module.

    :params filters: number of convolution kernels
    :params kernel_size: tuple, the kernel dimensions
    """
    return Conv2D(filters=filters,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  padding="same",
                  activation=relu)

def batch_norm():
    """Shorthand for a batchnorm layer."""
    return BatchNormalization(momentum=0.01, center=False)

def max_pool():
    """Max pooling layer."""

    return MaxPooling2D(pool_size=(2, 2),
                        strides=(2, 2),
                        padding="valid")

def upconv_relu(filters):
    """Layer. Short for batch norm, conv, relu"""

    return Deconv2D(filters=filters,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding="same",
                    activation=relu)

def convout():
    "Output layer."

    return Conv2D(filters=1,
                  kernel_size=(1, 1),
                  padding="same",
                  activation=sigmoid)

def build_unet(input_shape, layers=4, base_num_filters=64):
    """Build a u-shaped convolutionnal network for object segmentation.

    :params input_shape: the shape of the input
    :params layers: , number of layers
    :params base_num_filters: int (default=64), the number of convolutionnal filters per layer

    :type input_shape: tupple of int
    :type layers: int (default=4)
    :type base_num_filters: int (default=64)

    :returns model: a Keras Model"""

    assert not input_shape[0] % 2**layers,\
            "Too many layers for input_size"

    print("Building model.")

    #Model
    inputs = Input(shape=input_shape)

    conv1 = conv_relu(base_num_filters)(inputs)

    net = []
    net.append(conv1)
    concat_layers = []

    # Downward part of the net
    for layer in range(layers):

        batch_norm1 = batch_norm()(net[-1])
        conv1 = conv_relu(base_num_filters)(batch_norm1)
        batch_norm2 = batch_norm()(conv1)
        conv2 = conv_relu(base_num_filters)(batch_norm2)
        batch_norm3 = batch_norm()(conv2)
        conv3 = conv_relu(base_num_filters)(batch_norm3)
        maxpool = max_pool()(conv3)

        net.append(maxpool)
        concat_layers.append(conv2)

    # Middle layer
    batch_norm1 = batch_norm()(net[-1])
    conv1 = conv_relu(base_num_filters)(batch_norm1)
    batch_norm2 = batch_norm()(conv1)
    conv2 = conv_relu(base_num_filters)(batch_norm2)
    batch_norm3 = batch_norm()(conv2)
    upconv = upconv_relu(base_num_filters)(batch_norm3)

    net.append(upconv)

    # Upward part of the net
    for layer in range(layers-1, 0, -1):

        concat = Concatenate()([concat_layers[layer], net[-1]])
        batch_norm1 = batch_norm()(concat)
        conv1 = conv_relu(base_num_filters)(batch_norm1)
        batch_norm2 = batch_norm()(conv1)
        conv2 = conv_relu(base_num_filters)(batch_norm2)
        batch_norm3 = batch_norm()(conv2)
        upconv = upconv_relu(base_num_filters)(batch_norm3)

        net.append(upconv)

    concat = Concatenate()([concat_layers[0], net[-1]])
    batch_norm1 = batch_norm()(concat)
    conv1 = conv_relu(base_num_filters)(batch_norm1)
    batch_norm2 = batch_norm()(conv1)
    conv2 = conv_relu(base_num_filters)(batch_norm2)
    outputs = convout()(conv2)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
