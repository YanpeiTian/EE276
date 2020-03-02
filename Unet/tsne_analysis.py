import os
import pdb
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator,save_img
import tensorflow as tf
from unet import U_Net
from unet_strip import U_Net_In, U_Net_Out,AutoEncoder
import numpy as np
import pandas as pd
import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# from conv_models_cifar import conv_model, mixed_model, DQN_model

BATCH_SIZE = 64
EPOCH_NUM = 1
MAX_ITER = 1000

def main():
    # data loading
    save_dir = os.path.join(os.getcwd(), 'saved_models')

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    x_test /=255
    x_train /=255

    # # Raw data analysis
    # x_test=x_test.reshape((-1,32*32*3))
    # feat_cols = [ 'pixel'+str(i) for i in range(x_test.shape[1]) ]
    # df = pd.DataFrame(x_test,columns=feat_cols)
    # df['y'] = y_test
    # print('Size of the dataframe: {}'.format(df.shape))
    #
    #
    # tsne_data=df[feat_cols].values
    # time_start = time.time()
    # tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    # tsne_results = tsne.fit_transform(tsne_data)
    # print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    #
    # df['tsne-2d-one'] = tsne_results[:,0]
    # df['tsne-2d-two'] = tsne_results[:,1]
    #
    # plt.figure(figsize=(16,10))
    # sns.scatterplot(
    #     x="tsne-2d-one", y="tsne-2d-two",
    #     hue="y",
    #     palette=sns.color_palette("hls", 10),
    #     data=df,
    #     legend="full",
    #     alpha=0.3
    # )
    # plt.show()

    # model
    ae=AutoEncoder()

    ae.compile(optimizer='adam', loss='mean_squared_error')
    ae.fit(x_train[:100], x_train[:100], epochs=EPOCH_NUM, batch_size=BATCH_SIZE, shuffle=True, validation_data=(x_test, x_test))
    #
    ae.summary()
    # ae.encoder.save_weights(os.path.join(save_dir, 'encoder.h5'))
    # ae.decoder.save_weights(os.path.join(save_dir, 'decoder.h5'))

    # Encoded Analysis
    ae.encoder.load_weights('./deep_l2/encoder.h5')
    ae.decoder.load_weights('./deep_l2/decoder.h5')
    encoded_imgs = ae.encoder(x_test)

    tsne_data=tf.keras.backend.eval(encoded_imgs).reshape((10000,-1))
    feat_cols = [ 'pixel'+str(i) for i in range(tsne_data.shape[1]) ]
    df = pd.DataFrame(tsne_data,columns=feat_cols)
    df['y'] = y_test
    print('Size of the dataframe: {}'.format(df.shape))

    # tsne_data=df[feat_cols].values
    tsne_data=df.values
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(tsne_data)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]

    plt.figure(figsize=(16,10))
    plt.xlim((-10,10 ))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=df,
        legend="full",
        alpha=0.3
    )
    plt.show()

    # decoded_imgs = ae.decoder(encoded_imgs)
    # for i in range(100):
    #     save_img('data/'+str(i)+'.png',x_test[i])
    #     temp=tf.keras.backend.eval(decoded_imgs[i])
    #     save_img('data/'+str(i)+'re.png',temp)
    # print("finish")

if __name__ == '__main__':
    main()
