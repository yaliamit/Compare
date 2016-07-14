#from __future__ import print_function


import numpy as np
import lasagne
import compare_net
import theano
import theano.tensor as T
import os



def np_standardize(input):


    s=np.std(input,axis=2,keepdims=True)#.reshape((n0,1)),reps=n1)
    m=np.mean(input,axis=2,keepdims=True)

    output=(input-m)/s
    return np.squeeze(output)


def np_correlation(input1,input2):

    n=np.shape(input1)
    n0=n[0]
    n1=n[1]

    s0=np.std(input1,axis=1,keepdims=True)#.reshape((n0,1)),reps=n1)
    s1=np.std(input2,axis=1,keepdims=True)#.reshape((n0,1)),reps=n1)
    m0=np.mean(input1,axis=1,keepdims=True)
    m1=np.mean(input2,axis=1,keepdims=True)

    corr=np.sum(((input1-m0)/s0)*((input2-m1)/s1), axis=1)/n1

    corr=(corr+np.float32(1.))/np.float32(2.)
    #corr=np.squeeze(corr)z
    return corr

def get_shifted_correlations(input_std):

    num=input_std.shape[1]
    leng=input_std.shape[-1]
    vdim=input_std.shape[-2]
    dim=input_std.shape[2]
    sr=2
    corrs=np.zeros((num,leng,2*sr+1))
    for l in range(leng):
        # Loop over range of possible shifts
        for ll in np.arange(l-sr,l+sr+1):
            if (ll>=0 and ll<leng):
                tcor=np.sum(input_std[0,...,l]*input_std[1,...,ll],axis=1)/dim
                # Add correlations vertically at same horizontal location for dp
                if (tcor.ndim>1):
                    corrs[:,l,ll-l+sr]=np.sum(tcor,axis=1)
                else:
                    corrs[:,l,ll-l+sr]=tcor
    corrs=(corrs+np.float32(1.))/np.float32(2.)
    return(corrs)

def optimize_dp(corrs):

    jump=1
    # Original range of shifts for correlation computation
    sr=2
    # Current search range for optimizing must be less than sr
    srr=2
    num=corrs.shape[0]
    leng=corrs.shape[1]
    nsr=corrs.shape[2]
    table_state=-np.ones((num,leng,nsr))
    table_cost=-10000*np.ones((num,leng,nsr))
    table_cost[:,0,]=corrs[:,0,]
    for l in np.arange(jump,leng,jump):
        prel=l-jump
        # For each state (shift) of l find best allowable (shift) state of l-1
        for s in np.arange(l-srr,l+srr+1,jump):
            if (s>=0 and s<leng):
                lowt=np.maximum(prel-srr,0);
                # Can't use a matching location that comes before the matching location of the previous step
                hight=np.minimum(prel+srr+1,s+1)
                if (hight>lowt):
                    iit=np.arange(lowt,hight,jump)-prel+sr
                    curr=np.max(table_cost[:,prel,iit],axis=1)
                    tcurr=np.argmax(corrs[:,prel,iit],axis=1)
                    table_state[:,l,s-l+sr]=tcurr+lowt-prel+sr
                    table_cost[:,l,s-l+sr]=corrs[:,l,s-l+sr]+curr



    maxc=np.max(table_cost[:,-1,],axis=1)

    return(maxc)

def run_network_on_image():

    import make_seqs
    ims1, ims1a, ims2=make_seqs.make_seqs(slength=6,num_seqs=1000)

    input_var1 =  T.tensor4('inputs')
    input_var2 = T.tensor4('inputs_comp')

    network = compare_net.build_cnn_new_conv(input_var1, input_var2)
    if (os.path.isfile('net.npy')):
        spars=np.load('net.npy')
        lasagne.layers.set_all_param_values(network,spars)
    test_corr = lasagne.layers.get_output(network, deterministic=True)
    test_fn = theano.function([input_var1, input_var2], [test_corr])

    tcorr_same=test_fn(ims1,ims1a)
    tcorr_diff=test_fn(ims1,ims2)
    tt_same_std=np_standardize(tcorr_same[0])
    tt_diff_std=np_standardize(tcorr_diff[0])
    corrs_same=get_shifted_correlations(tt_same_std)
    corrs_diff=get_shifted_correlations(tt_diff_std)
    dps=optimize_dp(corrs_same)
    dpd=optimize_dp(corrs_diff)
    print(np.min(dps),np.max(dps),np.min(dpd),np.max(dpd))
    import pylab as py
    py.figure(1)
    py.hist(dps,alpha=.5)
    py.hist(dpd,alpha=.5)
    py.show()
    print('done ')


def run_network_on_all_pairs():

    import make_seqs
    num_seqs=10
    ims1, ims1a, ims2=make_seqs.make_seqs(slength=6,num_seqs=num_seqs, from_font=False)

    input_var1 =  T.tensor4('inputs')
    input_var2 = T.tensor4('inputs_comp')

    network = compare_net.build_cnn_new_conv(input_var1, input_var2)
    if (os.path.isfile('net.npy')):
        spars=np.load('net.npy')
        lasagne.layers.set_all_param_values(network,spars)
    test_corr = lasagne.layers.get_output(network, deterministic=True)
    test_fn = theano.function([input_var1, input_var2], [test_corr])

    tcorr_same=test_fn(ims1,ims1a)
    tt_same_std=np_standardize(tcorr_same[0])
    temp=np.copy(tt_same_std)
    ii=np.arange(0,num_seqs)
    np.random.shuffle(ii)
    iii=np.copy(ii)
    temp[1,]=tt_same_std[1,iii]
    dps=np.zeros((num_seqs,num_seqs))
    for n in range(num_seqs):
        temp[1,]=tt_same_std[1,np.roll(ii,-n),]
        corrs_same=get_shifted_correlations(temp)
        dps[n,]=optimize_dp(corrs_same)
    dps=dps.transpose()
    dpss=dps.copy()
    for n in range(num_seqs):
        dpss[n,]=np.roll(dps[n,],n)
    print(dpss)
    print(iii)
    print("done")
    dpss=np.max(dpss)-dpss

    #dps=dps.transpose()
    match_them(dpss,iii)
    print('done ')

def match_them(matrix,iii):
    from munkres import Munkres, print_matrix
    omatrix=np.copy(matrix)
    oii=np.argmin(omatrix,axis=1)

    m = Munkres()
    indexes = m.compute(matrix)
    jjj=np.zeros((len(iii),2))
    for i in range(len(iii)):
        jjj[iii[i],0]=iii[i]
        jjj[iii[i],1]=i
    #print_matrix(matrix, msg='Lowest cost through this matrix:')
    total = 0
    for i, (row, column) in enumerate(indexes):
        value = omatrix[row][column]
        total += value
        print '(%d, %d, %d, %d) -> %f' % (row, column, jjj[i,1], oii[i], value)
    print 'total cost: %d' % total
    error=np.double(np.sum(np.array(indexes)[:,1]!=jjj[:,1]))/np.double(len(iii))
    error_e=np.double(np.sum(np.array(indexes)[:,1]!=oii))/np.double(len(iii))
    print('ERROR',error, error_e)