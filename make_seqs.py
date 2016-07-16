import mnist
import numpy as np
import scipy.misc
import crop
import theano
import theano.tensor as T
import time
import lasagne
import os



def make_seqs(slength=2, num_seqs=20, from_font=False):

#def make_seqs(slength=4):
    imheight=48
    imwidth=48
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = mnist.load_dataset()
    if (from_font):
        Xfont=crop.get_fonts()
    begin=0
    incr=30
    lrange=3
    lrange2=lrange*2+1
    labels=np.floor(np.random.rand(num_seqs,slength,2)*10)
    TEST1=[]
    TEST1a=[]
    TEST2=[]
    for s in range(num_seqs):
        begin1=begin1a=begin2=np.int32(0)
        ii1=labels[s,:,0]
        ii2=labels[s,:,1]
        test1=np.zeros((imheight,imwidth*slength))
        test1a=np.zeros((imheight,imwidth*slength))
        test2=np.zeros((imheight,imwidth*slength))
        if (np.max(np.abs(ii1-ii2))>0):
            for k in range(slength):
                jj1=np.where(y_val==ii1[k])[0]
                jj2=np.where(y_val==ii2[k])[0]
                r1s=np.int32(np.floor(np.random.rand(2)*np.double(len(jj1))))
                r2=np.int32(np.floor(np.random.rand()*np.double(len(jj2))))
                sample1=scipy.misc.imresize(np.squeeze(X_val[jj1[r1s[0]],]),(imheight,imwidth))
                if (not from_font):
                    sample1a=np.squeeze(X_val[jj1[r1s[1]],])
                else:
                    sample1a=Xfont[np.int32(ii1[k]),]

                sample2=np.squeeze(X_val[jj2[r2],])
                sample1a=scipy.misc.imresize(sample1a,(imheight,imwidth))
                sample2=scipy.misc.imresize(sample2,(imheight,imwidth))
                test1[:,begin1:begin1+sample1.shape[1]]=np.maximum(sample1,test1[:,begin1:begin1+sample1.shape[1]])
                test1a[:,begin1a:begin1a+sample1.shape[1]]=np.maximum(sample1a,test1a[:,begin1a:begin1a+sample1.shape[1]])
                test2[:,begin2:begin2+sample1.shape[1]]=np.maximum(sample2,test2[:,begin2:begin2+sample1.shape[1]])
                begin1+=np.int32(np.floor(np.random.rand()*lrange2)-lrange+incr)
                begin1a+=np.int32(np.floor(np.random.rand()*lrange2)-lrange+incr)
                begin2+=np.int32(np.floor(np.random.rand()*lrange2)-lrange+incr)
        TEST1.append(test1)
        TEST1a.append(test1a)
        TEST2.append(test2)

    import pylab as py
    ii=range(num_seqs)
    np.random.shuffle(ii)
    for i in range(5):
        py.figure(num=1,figsize=(12,2),dpi=80)
        py.subplot(131)
        py.imshow(TEST1[ii[i]],aspect='equal')
        py.axis('off')
        py.subplot(132)
        py.imshow(TEST1a[ii[i]],aspect='equal')
        py.axis('off')
        py.subplot(133)
        py.imshow(TEST2[ii[i]],aspect='equal')
        py.axis('off')
        py.show()

    TEST1=np.float32(np.expand_dims(np.array(TEST1),1))
    TEST1a=np.float32(np.expand_dims(np.array(TEST1a),1))
    TEST2=np.float32(np.expand_dims(np.array(TEST2),1))

    return TEST1,TEST1a, TEST2