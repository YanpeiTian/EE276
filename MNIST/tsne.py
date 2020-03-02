from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model
from keras.preprocessing.image import save_img
import data
import dense_models, conv_models
import numpy as np
from keras.datasets import cifar10,mnist
import tensorflow.keras.backend as K
import pandas as pd
import time
from sklearn.manifold import TSNE
import seaborn as sns

import matplotlib.pyplot as plt

def main():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train=x_train.astype(np.float32)
    x_test=x_test.astype(np.float32)
    x_train=x_train/255
    x_test=x_test/255

    # Raw data
    # x_test=x_test.reshape((-1,28*28))
    # feat_cols = [ 'pixel'+str(i) for i in range(x_test.shape[1]) ]
    # df = pd.DataFrame(x_test,columns=feat_cols)
    # df['y'] = y_test
    # print('Size of the dataframe: {}'.format(df.shape))
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

    # print(x_train.shape)
    # print(x_test.shape)

    # model,encoder,decoder=dense_models.deep_model()
    encoder=load_model('models/encoder.h5')
    decoder=load_model('models/decoder.h5')

    x_train=K.constant(x_train.reshape((-1,28*28)))
    x_test=K.constant(x_test.reshape((-1,28*28)))

    encoded_imgs = encoder(x_test)

    # Encoded
    # feat_cols = [ 'pixel'+str(i) for i in range(encoded_imgs.shape[1]) ]
    # df = pd.DataFrame(K.eval(encoded_imgs),columns=feat_cols)
    # df['y'] = y_test
    # print('Size of the dataframe: {}'.format(df.shape))
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


    decoded_imgs = decoder(encoded_imgs)

    feat_cols = [ 'pixel'+str(i) for i in range(decoded_imgs.shape[1]) ]
    df = pd.DataFrame(K.eval(decoded_imgs),columns=feat_cols)
    df['y'] = y_test
    print('Size of the dataframe: {}'.format(df.shape))

    tsne_data=df[feat_cols].values
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(tsne_data)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]

    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=df,
        legend="full",
        alpha=0.3
    )
    plt.show()
    # x_test=K.eval(x_test).reshape((-1,28,28,1))
    # decoded_imgs=K.eval(decoded_imgs).reshape((-1,28,28,1))
    # for i in range(100):
    #     save_img('data/'+str(i)+'.png',x_test[i])
    #     # temp=tf.keras.backend.eval(decoded_imgs[i])
    #     save_img('data/'+str(i)+'re.png',decoded_imgs[i])
    # print("finish")

if __name__ == "__main__":
    main()
