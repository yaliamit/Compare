
import numpy as np
import theano
import theano.tensor as T
import time
import lasagne



class CorrLayer(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        input1=input[0,]
        input2=input[1,]
        n=self.input_shape
        #n0=n[1]
        n1=n[2]
        # tt=tuple([n0,1])
        s0=T.std(input1,axis=1,keepdims=True)
        s1=T.std(input2,axis=1,keepdims=True)
        m0=T.mean(input1,axis=1,keepdims=True)
        m1=T.mean(input2,axis=1,keepdims=True)


        corr=T.sum(((input1-m0)/s0)*((input2-m1)/s1), axis=1)/n1

        corr=(corr+np.float32(1.))/np.float32(2.)
        return corr


    def get_output_shape_for(self, input_shape):
        return (tuple((input_shape[1], 1, input_shape[3], input.shape[4])))