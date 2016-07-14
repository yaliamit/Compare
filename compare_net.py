from __future__ import print_function

import numpy as np
import lasagne

def build_cnn_new(input_var1=None,input_var2=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    input = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var1)
    input_comp=lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var2)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    conv2d1 = lasagne.layers.Conv2DLayer(
            input, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(),name="conv1")

    # Max-pooling layer of factor 2 in both dimensions:
    pool2d1 = lasagne.layers.MaxPool2DLayer(conv2d1, pool_size=(2, 2))

    conv2d2 = lasagne.layers.Conv2DLayer(
            pool2d1, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(),name="conv2")

    # Max-pooling layer of factor 2 in both dimensions:
    pool2d2 = lasagne.layers.MaxPool2DLayer(conv2d2, pool_size=(2, 2))

    dense256 = lasagne.layers.DenseLayer(
            pool2d2,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,name="dense1")

    pars=lasagne.layers.get_all_params(dense256,trainable=True)


    conv2d1_comp = lasagne.layers.Conv2DLayer(
            input_comp, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=conv2d1.W, b=conv2d1.b)

# Max-pooling layer of factor 2 in both dimensions:
    pool2d1_comp = lasagne.layers.MaxPool2DLayer(conv2d1_comp, pool_size=(2, 2))

    conv2d2_comp = lasagne.layers.Conv2DLayer(
            pool2d1_comp, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=conv2d2.W, b= conv2d2.b)

    # Max-pooling layer of factor 2 in both dimensions:
    pool2d2_comp = lasagne.layers.MaxPool2DLayer(conv2d2_comp, pool_size=(2, 2))

    dense256_comp = lasagne.layers.DenseLayer(
            pool2d2_comp,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify, W=dense256.W, b=dense256.b)


    d256=lasagne.layers.dimshuffle(dense256,('x',0,1))
    d256_c=lasagne.layers.dimshuffle(dense256_comp,('x',0,1))
    final=lasagne.layers.ConcatLayer((d256,d256_c),0)

    return final

def build_cnn_new_conv(input_var1=None,input_var2=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    input = lasagne.layers.InputLayer(shape=(None, 1, None, None),
                                        input_var=input_var1)
    input_comp=lasagne.layers.InputLayer(shape=(None, 1, None, None),
                                        input_var=input_var2)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    conv2d1 = lasagne.layers.Conv2DLayer(
            input, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(),name="conv1")

    # Max-pooling layer of factor 2 in both dimensions:
    pool2d1 = lasagne.layers.MaxPool2DLayer(conv2d1, pool_size=(2, 2))

    conv2d2 = lasagne.layers.Conv2DLayer(
            pool2d1, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(),name="conv2")

    # Max-pooling layer of factor 2 in both dimensions:
    pool2d2 = lasagne.layers.MaxPool2DLayer(conv2d2, pool_size=(2, 2))

    conv2d3 = lasagne.layers.Conv2DLayer(
            pool2d2,
            num_filters=256,filter_size=(4,4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(),name="conv2d3")


    conv2d1_comp = lasagne.layers.Conv2DLayer(
            input_comp, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=conv2d1.W, b=conv2d1.b)

# Max-pooling layer of factor 2 in both dimensions:
    pool2d1_comp = lasagne.layers.MaxPool2DLayer(conv2d1_comp, pool_size=(2, 2))

    conv2d2_comp = lasagne.layers.Conv2DLayer(
            pool2d1_comp, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=conv2d2.W, b= conv2d2.b)

    # Max-pooling layer of factor 2 in both dimensions:
    pool2d2_comp = lasagne.layers.MaxPool2DLayer(conv2d2_comp, pool_size=(2, 2))

    conv2d3_comp = lasagne.layers.Conv2DLayer(
            pool2d2_comp,
            num_filters=256,filter_size=(4,4),
            nonlinearity=lasagne.nonlinearities.rectify,W=conv2d3.W, b=conv2d3.b)


    #conv2d3=lasagne.layers.flatten(conv2d3)
    #conv2d3_comp=lasagne.layers.flatten(conv2d3_comp)
    conv2d3=lasagne.layers.dimshuffle(conv2d3,('x',0,1,2,3))
    conv2d3_comp=lasagne.layers.dimshuffle(conv2d3_comp,('x',0,1,2,3))
    final=lasagne.layers.ConcatLayer((conv2d3,conv2d3_comp),0)


    return final


def build_cnn_deep(input_var1=None,input_var2=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    input = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var1)
    input_c=lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var2)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    conv2d1 = lasagne.layers.Conv2DLayer(
            input, num_filters=32, filter_size=(3, 3),pad=1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(),name="conv1")
    conv2d1_1 = lasagne.layers.Conv2DLayer(
            conv2d1, num_filters=32, filter_size=(3, 3),pad=1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(),name="conv1.1")
    # Max-pooling layer of factor 2 in both dimensions:
    pool2d1 = lasagne.layers.MaxPool2DLayer(conv2d1_1, pool_size=(2, 2))

    conv2d2 = lasagne.layers.Conv2DLayer(
            pool2d1, num_filters=32, filter_size=(3, 3),pad=1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(),name="conv2")
    conv2d2_1 = lasagne.layers.Conv2DLayer(
            conv2d2, num_filters=32, filter_size=(3, 3),pad=1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(),name="conv2.2")
    # Max-pooling layer of factor 2 in both dimensions:
    pool2d2 = lasagne.layers.MaxPool2DLayer(conv2d2_1, pool_size=(2, 2))

    dense256 = lasagne.layers.DenseLayer(
            pool2d2,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,name="dense1")

    #pars=lasagne.layers.get_all_params(dense256,trainable=True)

    conv2d1_c = lasagne.layers.Conv2DLayer(
            input_c, num_filters=32, filter_size=(3, 3),pad=1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=conv2d1.W, b= conv2d1.b, name="conv1")
    conv2d1_1_c = lasagne.layers.Conv2DLayer(
            conv2d1_c, num_filters=32, filter_size=(3, 3),pad=1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=conv2d1_1.W, b=conv2d1_1.b, name="conv1.1")
    # Max-pooling layer of factor 2 in both dimensions:
    pool2d1_c = lasagne.layers.MaxPool2DLayer(conv2d1_1_c, pool_size=(2, 2))

    conv2d2_c = lasagne.layers.Conv2DLayer(
            pool2d1_c, num_filters=32, filter_size=(3, 3),pad=1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=conv2d2.W, b=conv2d2.b, name="conv2")
    conv2d2_1_c = lasagne.layers.Conv2DLayer(
            conv2d2_c, num_filters=32, filter_size=(3, 3),pad=1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=conv2d2_1.W, b=conv2d2_1.b,name="conv2.2")
    # Max-pooling layer of factor 2 in both dimensions:
    pool2d2_c = lasagne.layers.MaxPool2DLayer(conv2d2_c, pool_size=(2, 2))

    dense256_c = lasagne.layers.DenseLayer(
            pool2d2_c,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify, W=dense256.W, b=dense256.b, name="dense1")



    d256=lasagne.layers.dimshuffle(dense256,('x',0,1))
    d256_c=lasagne.layers.dimshuffle(dense256_c,('x',0,1))
    final=lasagne.layers.ConcatLayer((d256,d256_c),0)

    return final


def create_paired_data_set(X,y,num):

    if (2*num> X.shape[0]):
        num=X.shape[0]/2
    print(num)
    ii=range(X.shape[0])
    np.random.shuffle(ii)
    ii1=ii[0:num]
    ii2=ii[num:(num+num)]
    ytr=y[ii1]==y[ii2]
    Xtr=X[ii1,]
    Xtr_comp=X[ii2,]
    return np.float32(Xtr), np.float32(Xtr_comp), np.float32(ytr)



def create_paired_data_set_with_fonts(X,y,num):
    import crop
    Xfont=crop.get_fonts()
    if (num> X.shape[0]):
        num=X.shape[0]
    print(num)
    ii=range(X.shape[0])
    np.random.shuffle(ii)
    ii1=ii[0:num]

    Xtr=np.repeat(X[ii1,],10,axis=0)
    yy=np.repeat(y[ii1],10)
    yyy=np.tile(range(10),num)
    Xtr_comp=Xfont[yyy,]
    ytr=(yy==yyy)
    ylab=y[ii1]
    Xtr_comp=np.expand_dims(Xtr_comp,axis=1)
    return np.float32(Xtr), np.float32(Xtr_comp), np.float32(ytr), np.int32(ylab)