
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import ch02 as api

if __name__ == '__main__':
    
    useTensorFlowNMIST = False
    
    # create an mlp object
    mlp = api.NeuralNetMLP(n_hidden=256, 
                           l2=0.01, 
                           epochs=20, 
                           eta=0.002, 
                           minibatch_size=100, 
                           shuffle=True, 
                           seed=1)
    
    # load the MNIST data
    train_x, train_y = [],[]
    
    if useTensorFlowNMIST:
        # load the MNIST data from tensorflow
        train_x, train_y = tf.keras.datasets.mnist.load_data()[0]
        test_x, test_y = tf.keras.datasets.mnist.load_data()[1]
        
        # reshape the data
        train_x, train_y = train_x.reshape(-1, 784), train_y
        test_x, test_y = test_x.reshape(-1, 784), test_y
    else:
        current_dir = pathlib.Path(__file__).parent.absolute()
        train_x, train_y = api.load_mnist(current_dir, kind='train')
        test_x, test_y = api.load_mnist(current_dir, kind='t10k')
    
    # fit an MLP classifier
    mlp.fit(train_x, train_y, test_x, test_y)
    
    # save the model as an .npz file
    np.savez_compressed('MLP.npz', w_h=mlp.w_h, b_h=mlp.b_h, w_out=mlp.w_out, b_out=mlp.b_out)
    
    # load the model as mlp_load 
    mlp_load = np.load('MLP.npz')
    
    # print the dimensions of each components of mlp_load, just to ensure everything is intact
    print(mlp_load['w_h'].shape)
    print(mlp_load['b_h'].shape)
    print(mlp_load['w_out'].shape)
    print(mlp_load['b_out'].shape)
    
    # print the accuracy of the model on the test set
    y_pred = mlp.predict(test_x)
    print('Test accuracy: %.2f%%' % ((np.sum(test_y == y_pred)).astype(np.float32) / test_x.shape[0] * 100))
    
    # plot the costs for the last layer and the train and test accuracy with subplots
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    ax[0].plot(range(mlp.epochs), mlp.eval_['cost'])
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Cost')
    ax[0].set_title('Cost')
    ax[1].plot(range(mlp.epochs), mlp.eval_['train_acc'], label='Training')
    ax[1].plot(range(mlp.epochs), mlp.eval_['valid_acc'], label='Validation')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Accuracy')
    ax[1].legend()
    plt.tight_layout()
    plt.show()
    
    