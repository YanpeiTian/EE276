import os
import pdb
import keras
import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

from conv_models_cifar import conv_model, mixed_model, DQN_model
from unet import U_Net

# data loading
BATCH_SIZE = 128 
EPOCH_NUM = 100
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# model
#autoencoder, encoder, decoder = conv_model()
unet = U_Net()

optimizer = tf.keras.optimizers.Adam(1e-5)
#optimizer = keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, amsgrad=False)
unet.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())
unet.fit(x_train, x_train, epochs=EPOCH_NUM, batch_size=BATCH_SIZE, shuffle=True, validation_data=(x_test, x_test))

unet.save(os.path.join(save_dir, 'unet.h5'))
# encoder.save(os.path.join(save_dir, 'encoder.h5'))
# decoder.save(os.path.join(save_dir, 'decoder.h5'))

if __name__ == '__main__':
    main()
