"""Let's train our model.

:Date: 2017-05-24
:Version: 1
:Author: Olivier Dolle"""

import argparse
import numpy as np

from keras.callbacks import TensorBoard, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

from net import build_unet
from generate_fake_data import generate_fake_data

def main(test, train_path, batch_size, epochs):
    """Entrypoint."""

    if test:
        print("Creating data.")
        data, labels = gen_fake_data(n_samples=100)
    else:
        data_generator = ImageDataGenerator(samplewise_center=True,
                                            samplewise_std_normalization=True,
                                            rotation_range=90)

    net = build_unet(input_shape=data.shape[1:])

    # Create tensorboard callback
    tb_callback = TensorBoard(log_dir='./graph',
                              histogram_freq=0,
                              write_graph=True,
                              write_images=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    print("Training.")
    if test:
        net.fit(data,
                labels,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[tb_callback, early_stopping],
                validation_split=0.2)
    else:
        net.fit_generator(data_generator.flow_from_directory(train_path, batch_size=batch_size),
                          steps_per_epoch=1000 // batch_size,
                          epochs=epochs,
                          callbacks=[tb_callback, early_stopping])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a u-net on data.")
    parser.add_argument('--train_path', dest='train_path', type=str,
                        help='Path to train_data in ??? format.')
    parser.add_argument('--epochs', dest='epochs', type=int, default=10,
                        help='Number of epochs the model will train on.')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1,
                        help="How much data should be fed at once.")
    parser.add_argument('--test', dest='test', action='store_const',
                        const=True, default=False,
                        help="Test the model against made-up data.")

    args = parser.parse_args()
    if args.train_path is not None:
        main(**vars(args))
    else:
        if args.test:
            main(**vars(args))
        else:
            parser.error("require --train_path"\
                        "or you can launch a test with --test")
