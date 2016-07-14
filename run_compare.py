import mnist
import compare_net
import numpy as np
import theano
import theano.tensor as T
import time
import lasagne
import os
import compare


def correlation(input1,input2):

    n=T.shape(input1)
    n0=n[0]
    n1=n[1]

    s0=T.std(input1,axis=1,keepdims=True)#.reshape((n0,1)),reps=n1)
    s1=T.std(input2,axis=1,keepdims=True)#.reshape((n0,1)),reps=n1)
    m0=T.mean(input1,axis=1,keepdims=True)
    m1=T.mean(input2,axis=1,keepdims=True)

    corr=T.sum(((input1-m0)/s0)*((input2-m1)/s1), axis=1)/n1

    corr=(corr+np.float32(1.))/np.float32(2.)
    corr=T.reshape(corr,(n0,))
    return corr

def iterate_minibatches_new(inputs1, inputs2, targets, batchsize, shuffle=False):
    assert len(inputs1) == len(targets) and len(inputs2)==len(targets)
    if shuffle:
        indices = np.arange(len(inputs1))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs1) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs1[excerpt], inputs2[excerpt], targets[excerpt]


def main_new(num_epochs=500, num_train=0, use_existing=False, rotate_angle=0):
    # Load the dataset
    batch_size=100
    thresh=.9
    eta_init=np.float32(.001)
    print("Loading data...")
    X_train_in, y_train_in, X_val_in, y_val_in, X_test_in, y_test_in = mnist.load_dataset()
    if (rotate_angle>0):
        X_train_in=mnist.rotate_dataset(X_train_in,angle=rotate_angle)
        X_val_in=mnist.rotate_dataset(X_val_in,angle=rotate_angle)
        X_test_in=mnist.rotate_dataset(X_test_in,angle=rotate_angle)

    if (num_train==0):
        num_train=np.shape(y_train_in)[0]
    #X_train_r=rotate_dataset(X_train,12,num_train)
    #X_val_r=rotate_dataset(X_val,12,np.shape(X_val)[0])
    X_train,  X_train_c, y_train=compare_net.create_paired_data_set(X_train_in, y_train_in, num_train)
    X_val, X_val_c, y_val = compare_net.create_paired_data_set(X_val_in, y_val_in, num_train)
    X_test, X_test_c, y_test = compare_net.create_paired_data_set(X_test_in, y_test_in, num_train)
    X_test1, X_test_f, y_test_f, y_label = compare_net.create_paired_data_set_with_fonts(X_test_in, y_test_in, 10000)

    # Prepare Theano variables for inputs and targets
    input_var1 =  T.tensor4('inputs')
    input_var2 = T.tensor4('inputs_comp')
    target_var = T.fvector('target')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")

    network = compare_net.build_cnn_new_conv(input_var1, input_var2)
    if (os.path.isfile('net.npy') and use_existing):
        spars=np.load('net.npy')
        lasagne.layers.set_all_param_values(network,spars)
        #layers=lasagne.layers.get_all_layers(network)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    corr = lasagne.layers.get_output(network)


    corr=correlation(corr[0,],corr[1,])
    #loss=T.mean(T.square(T.sum(corr,axis=1)-target_var))
    loss=T.mean(T.square(corr-target_var))

    acc = T.mean(T.eq(corr>thresh, target_var),
                      dtype=theano.config.floatX)

    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    print(params)

    eta = theano.shared(np.array(eta_init, dtype=theano.config.floatX))
    #eta_decay = np.array(0.95, dtype=theano.config.floatX)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=eta, momentum=0.9)
    #updates = lasagne.updates.sgd(
    #        loss, params, learning_rate=eta)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_corr = lasagne.layers.get_output(network, deterministic=True)
    test_corr=correlation(test_corr[0,], test_corr[1,])
    #test_loss=T.mean(T.square(T.sum(test_corr,axis=1)-target_var))
    test_loss=T.mean(T.square(test_corr-target_var))

    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(test_corr>thresh, target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var1, input_var2, target_var], [loss, acc, corr], updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var1, input_var2, target_var], [test_loss, test_acc, test_corr])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    t=1
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_acc = 0
        train_batches = 0
        start_time = time.time()
        print(eta.get_value())
        for batch in iterate_minibatches_new(X_train,X_train_c, y_train, batch_size, shuffle=True):
            inputs1, inputs2, targets = batch
            eta.set_value(eta_init) #/np.float32(t))
            bloss, bacc, bcorr = train_fn(inputs1,inputs2,targets)

            train_err += bloss
            train_acc += bacc
            train_batches += 1
            t=t+1

        # And a full pass over the validation data:

        val_acc=0
        val_err = 0
        val_batches = 0
        for batch in iterate_minibatches_new(X_val,X_val_c, y_val, batch_size, shuffle=False):
            inputs1, inputs2, targets = batch
            err, acc, tcorr = val_fn(inputs1, inputs2, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print(" train accuracy:\t\t{:.6f}".format(train_acc/ train_batches))

        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print(" validation accuracy:\t\t{:.6f}".format(val_acc/ val_batches))

        if (np.mod(epoch,10)==0 and epoch>0):
            params = lasagne.layers.get_all_param_values(network)
            np.save('net',params)

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0

    for batch in iterate_minibatches_new(X_test, X_test_c, y_test, batch_size, shuffle=False):
        inputs1, inputs2, targets = batch
        err, acc, tcorr = val_fn(inputs1, inputs2, targets)

        test_acc += acc
        test_err += err

        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test acc:\t\t\t{:.6f}".format(test_acc / test_batches))


    try:
        X_test1
    except NameError:
       print "X_test1 not defined"
    else:
        test_err = 0
        test_acc = 0
        test_batches = 0
        corrs=[]
        for batch in iterate_minibatches_new(X_test1, X_test_f, y_test_f, batch_size, shuffle=False):
            inputs1, inputs2, targets = batch
            err, acc, tcorr = val_fn(inputs1, inputs2, targets)

            corrs.append(np.reshape(tcorr,(10,-1)))
            test_acc += acc
            test_err += err
            test_batches += 1

        CORRS=np.vstack(corrs)
        yii=np.argmax(CORRS,axis=1)
        print("Final results classification:")
        print("  test loss font:\t\t\t{:.6f}".format(test_err / test_batches))
        print("  test acc font:\t\t\t{:.6f}".format(np.double(np.sum(yii==y_label)) / len(yii)))


#main_new(num_epochs=0,num_train=20000,use_existing=True, rotate_angle=20)
#compare.run_network_on_image()
compare.run_network_on_all_pairs()