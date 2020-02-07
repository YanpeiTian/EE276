import reconstruct
import conv_models
import numpy as np

MINI_BATCH=100
MAX_ITER=1
EPOCH = 50

def main():

    # Define the model, model currently following the DQN paper
    autoencoder,encoder,decoder=conv_models.conv_model()

    for i in range(MAX_ITER):
        print('Randomly select a mini batch for training...')
        minibatch=reconstruct.generate_data(MINI_BATCH)
        print('Loading training data, Done')

        print('Processing the minibatch for training...')
        x_train=[]
        for img in minibatch:
            patchsize=img['patch_size']
            x_train.append(np.asarray(img['patches']).reshape((-1,patchsize,patchsize,3)))

        x_train=np.concatenate(x_train)
        x_test=x_train[-100:]
        x_train=x_train[:-100]
        print('Processing the minibatch, Done')

        print('Training a mini-batch...')
        autoencoder.fit(x_train, x_train,epochs=EPOCH,batch_size=256,shuffle=True,validation_data=(x_test, x_test))
        print('Training, Done')

    print('Saving model...')
    encoder.save('encoder.h5')
    decoder.save('decoder.h5')


if __name__ == "__main__":
    main()
