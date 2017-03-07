import theano
import theano.tensor as T
import numpy as np
import os
import theano.tensor.signal.pool as pool

#random number generator
rng = np.random.RandomState(23433)

def init_weight(Layer_num, w_shp):
    path = r'./para/Layer_'+str(Layer_num)+'_weight.npy'
    w_bound = np.sqrt(w_shp[1] * w_shp[2] * w_shp[3])
    if os.path.exists(path):
        W_value = np.load(path)
        print 'load '+'Layer_'+str(Layer_num)+'_weight.npy'
    else:
        W_value = np.array(
                            rng.uniform(-1.0/w_bound,
                            1.0/w_bound,
                            size = w_shp),
                            dtype='float64')
        print 'random init '+'Layer_'+str(Layer_num)+'_weight'
    return W_value

def init_B(Layer_num, b_shp):
    path = r'./para/Layer_'+str(Layer_num)+'_bias.npy'
    if os.path.exists(path):
        B_value = np.load(path)
        print 'load '+'Layer_'+str(Layer_num)+'_bias.npy'
    else:
        B_value = np.array(
                            rng.uniform(-0.0,
                            0.0,
                            size = b_shp),
                            dtype='float64')
        print 'random init '+'Layer_'+str(Layer_num)+'_bias'
    return B_value


class Conv_Layer:
    def __init__(self, Layer_num, layer_input,
                 mini_batch_size, input_feature_num, img_shp,
                 this_layer_feature_num, last_layer_feature_num, filter_shp,
                 pool_shp
                 ):
        self.num = Layer_num
        #function conv2d(input, W),ip_shp = input.shape(); w_shp = W.shape()
        self.ip_shp = (mini_batch_size, input_feature_num, img_shp[0], img_shp[1])
        self.w_shp = (this_layer_feature_num, last_layer_feature_num, filter_shp[0], filter_shp[1])

        #input is a tensor4, input's shape = ip_shp
        self.input = layer_input

        #weight is a shared type, it saves the filters, its shape = w_shp
        self.W = theano.shared(init_weight(Layer_num, self.w_shp), 'Layer_'+str(Layer_num)+'_weight')

        #bias of this layer, size depend on the number of layer's feature maps
        b_shp = (this_layer_feature_num, )
        self.B = theano.shared(init_B(Layer_num, b_shp), 'Layer_'+str(Layer_num)+'_bias')

        self.conv_result = T.nnet.conv2d(self.input, self.W)

        self.pool_result = pool.pool_2d(self.conv_result, ds = pool_shp, ignore_border=False)#shape(batch_size, layer0_feature_num, 14, 14)

        #output of this layer
        self.output = T.nnet.relu(self.pool_result + self.B.dimshuffle('x', 0, 'x', 'x'))

        #parament
        self.para = [self.W, self.B]




