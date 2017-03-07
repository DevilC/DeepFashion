import theano
import theano.tensor as T
import numpy as np
import os

rng = np.random.RandomState(23453)

def weight_init(layer_num, W_bound, shape):
    path = r'./para/Layer_'+str(layer_num)+'_weight.npy'
    if os.path.exists(path) == True:
        w = np.load(path)
        print 'load '+'Layer_'+str(layer_num)+'_weight.npy'
        return w
    else:
        w = rng.uniform(-W_bound,
                        W_bound,
                        size = shape)
        print 'random init '+'Layer_'+str(layer_num)+'_weight.npy'
        return w

def B_init(layer_num, B_bound, shape):
    path = r'./para/Layer_'+str(layer_num)+'_bias.npy'
    if os.path.exists(path):
        b = np.load(path)
        print 'load ' + 'Layer_' + str(layer_num) + '_bias.npy'
    else:
        b = rng.uniform(-B_bound,
                        B_bound,
                        size = shape)
        print 'random init '+'Layer_'+str(layer_num)+'_bias'
    return b

class layer():
    def __init__(self, layer_num, layer_input, input_num, out_num):
        self.num = layer_num
        self.input_num = input_num
        self.output_num = out_num
        self.input = layer_input

        W_bound = np.sqrt(float(out_num)/(input_num + out_num))
        self.W = theano.shared(weight_init(layer_num, W_bound, (input_num,out_num)), 'Layer_'+str(layer_num)+'_weight')

        b_value = B_init(layer_num, 0.0, (out_num,))
        self.B = theano.shared(b_value, 'Layer_'+str(layer_num)+'_bias')

        self.y = T.dot(self.input, self.W) + self.B.dimshuffle('x', 0)
        self.output = T.nnet.relu(self.y)

        self.para = [self.W, self.B]

class MLP:
    def __init__(self, MLP_input, input_num, hidden_num, output_num):
        self.layer0 = layer(2, MLP_input, input_num, hidden_num)

        self.layer1 = layer(3, self.layer0.output, hidden_num, output_num)

        self.output = T.nnet.softmax(self.layer1.y)

        self.para = self.layer0.para + self.layer1.para