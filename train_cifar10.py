import os
import pdb
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from unet import U_Net
from unet_strip import U_Net_In, U_Net_Out,AutoEncoder
import numpy as np
import matplotlib.pyplot as plt
from conv_models_cifar import conv_model, mixed_model, DQN_model
from keras.callbacks import TensorBoard

BATCH_SIZE = 64
EPOCH_NUM = 100 
MAX_ITER = 1000

def main():
    # data loading
    save_dir = os.path.join(os.getcwd(), 'saved_models')

    model_name = 'keras_cifar10_trained_model.h5'

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    x_test /=255
    x_train /=255

    # model
    autoencoder=AutoEncoder()

    def ssim_loss(img1,img2):
        return tf.image.ssim(img1, img2, max_val=1.0)

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=ssim_loss)
    # autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder.fit(x_train, x_train, epochs=EPOCH_NUM, batch_size=BATCH_SIZE, shuffle=True, validation_data=(x_test, x_test), callbacks=[TensorBoard(log_dir='.log/')])

    #
    autoencoder.encoder.save_weights(os.path.join(save_dir, 'encoder.h5'))
    autoencoder.decoder.save_weights(os.path.join(save_dir, 'decoder.h5'))

if __name__ == '__main__':
    main()
