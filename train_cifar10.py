import os
import pdb
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

from conv_models_cifar import conv_model, mixed_model, DQN_model

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

    return

    # model
    autoencoder, encoder, decoder = conv_model()

    def ssim_loss(img1,img2):
        return tf.image.ssim(img1, img2, max_val=1.0)

    autoencoder.compile(optimizer='adadelta', loss=ssim_loss)
    autoencoder.fit(x_train, x_train, epochs=EPOCH_NUM, batch_size=BATCH_SIZE, shuffle=True, validation_data=(x_test, x_test))

    encoder.save(os.path.join(save_dir, 'encoder.h5'))
    decoder.save(os.path.join(save_dir, 'decoder.h5'))

if __name__ == '__main__':
    main()
