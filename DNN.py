""" 
DNN is a little python module to demonstrate 
the basic elements of a deep neural network 
in action.
"""
import numpy as np
import matplotlib.pyplot as plt

from pointwise_activations import func_list
from loss_functions import loss_list
from Solvers import *

class Data(object):
    """
    Data class takes care of the minibatch management

    There are two optional parameters for Data objects.
    -- batch_size: if lefts as None will use the entire data set for 
       each pass.
    -- shuffle: wich defaults to true will shuffle data points to feed 
       the network at each iteration. When a epoch has been reached, 
       data is reshuffled.
    """
    batch_iter = None 
    data_size = None
    batch_idx = None
    crt_idx = None
    n_batches = None
    shuffle = None
    
    def __init__(self, data, batch_size=None, shuffle=True):
        self.data = data
        self.data_size = data.shape[0]
        self.shuffle = shuffle
        if batch_size is None:
            self.batch_size = self.data_size
            self.batch_iter = None
            self.batch_idx = None
            self.shuffle = False
        else:
            assert self.data_size >= batch_size, 'batch_size exeeds number of data samples'
            self.batch_size = batch_size
            self.batch_iter = 0
        self.n_batches = self.data_size / self.batch_size
        leftovers = self.data_size % self.batch_size
                
    def getBatch(self):
        if self.batch_iter is None:
            return self.data
  
        if self.batch_iter == 0:
            ## shuffle data at the begining of every epoch
            if self.shuffle is True:
                self.batch_idx = np.random.permutation(self.data_size) 
            else:
                self.batch_idx = np.arange(self.data_size)
            self.batch_idx = np.reshape(self.batch_idx, (self.n_batches, -1))
        self.crt_idx = self.batch_idx[self.batch_iter, :]
        batch = self.data[self.crt_idx, ::]
        self.batch_iter += 1
        if self.batch_iter == self.n_batches:
            self.batch_iter = 0
        return batch

    def getDataAsIn(self, ref_data_object=None):
        """
        This is an auxiliary method with the purpose of selecting
        the corresponding data-target pairs. 
        """
        if ref_data_object.crt_idx is not None:
            return self.data[ref_data_object.crt_idx, ::]
        else:
            return self.data

    
        
class Layer(object):
    """
    Layer class implements a uniform composition of affine map followed by 
    point-wise nonlinearity.
    
    Input: 
    -- n_in: number of inputs
    -- n_out: number of outputs (numer of units)
    -- activation: point-wise nonlinearity. 
       --logistic, 
       --tanh, 
       --relu, 
       --abs,
       --square, 
       --halfsquare
    """

    X0 = None
    X1 = None
    Z = None
    D0 = None
    D1 = None
    params = None
    params_aux = None
    deltas = None
    grads = None
    g = None
    g_prime = None
    n_in = None
    n_out = None

    def __init__(self, n_in, n_out, activation):
        assert n_in is not None and n_out is not None, "layer must hav valid inout output sizes"
        self.n_in = n_in
        self.n_out = n_out
        self.params = {}
        self.params_aux = {}
        self.deltas = {}
        self.grads = {}
        self.params['W'] = np.random.normal(size=(self.n_in, self.n_out)) / np.sqrt(self.n_in)
        self.params['b'] = np.zeros((1, self.n_out))
        for paramname in self.params.keys():
            self.params_aux[paramname] = np.zeros_like(self.params[paramname]) 
            self.deltas[paramname] = np.zeros_like(self.params[paramname]) 
            self.grads[paramname] = np.zeros_like(self.params[paramname])
        self.g = func_list[activation][0]
        self.g_prime = func_list[activation][1]
    
        
    def forward(self, aux=False):
        if aux is False:
            self.Z = np.dot(self.X0, self.params['W']) + self.params['b']
        else:
            self.Z = np.dot(self.X0, self.params_aux['W']) + self.params_aux['b']
        self.X1 = self.g(self.Z)
    
    def backward(self, aux=False):
        if self.D1 is None:
            G = self.g_prime(self.Z)
        else:
            G = np.multiply(self.D1, self.g_prime(self.Z))
        if aux is False:
            self.D0 = np.dot(G, self.params['W'].transpose())
        else:
            self.D0 = np.dot(G, self.params_aux['W'].transpose())
        self.grads['W'] = np.dot(self.X0.transpose(), G)
        self.grads['b'] = np.sum(G, axis=0)
            
    def updateParam(self, solver_func):
        solver_func(self)
        self.params['W'] += self.deltas['W']
        self.params['b'] += self.deltas['b']

class RNNLayer(object):
    """
    Layer class implements a uniform composition of affine map followed by 
    point-wise nonlinearity.
    
    Input: 
    -- n_in: number of inputs
    -- n_out: number of outputs (numer of units)
    -- activation: point-wise nonlinearity. 
       --logistic, 
       --tanh, 
       --relu, 
       --abs,
       --square, 
       --halfsquare
    """
    t_trunc = None
    X0 = None
    X1 = None
    H = None
    Z_h = None
    Z_o = None
    D0 = None
    DH = None
    D1 = None
    params = None
    params_aux = None
    deltas = None
    grads = None
    g_h = None
    g_h_prime = None
    g_o = None
    g_o_prime = None
    n_in = None
    n_out = None
    h_hid = None
    keep_hid = None

    def __init__(self, n_in, n_out, n_hid, hid_activation, out_activation, t_trunc=None):
        assert n_in is not None and n_out is not None and n_hid is not None, "layer must hav valid inout output sizes"
        self.n_in = n_in
        self.n_out = n_out
        self.n_hid = n_hid
        self.params = {}
        self.params_aux = {}
        self.deltas = {}
        self.grads = {}
        self.params['W_vh'] = np.random.normal(size=(self.n_in, self.n_hid)) / np.sqrt(self.n_in)
        self.params['b_h'] = np.zeros((1, self.n_hid))
        self.params['W_hh'] = 0.1*np.random.normal(size=(self.n_hid, self.n_hid)) / np.sqrt(self.n_hid)
        self.params['b_o'] = np.zeros((1, self.n_out))
        self.params['W_ho'] = np.random.normal(size=(self.n_hid, self.n_out)) / np.sqrt(self.n_hid)
        for paramname in self.params.keys():
            self.params_aux[paramname] = np.zeros_like(self.params[paramname]) 
            self.deltas[paramname] = np.zeros_like(self.params[paramname]) 
            self.grads[paramname] = np.zeros_like(self.params[paramname])
        self.g_h = func_list[hid_activation][0]
        self.g_h_prime = func_list[hid_activation][1]
        self.g_o = func_list[out_activation][0]
        self.g_o_prime = func_list[out_activation][1]
        self.keep_hid = False

    def keepHid(self, state):
        self.keep_hid = state
        
    def forward(self, aux=False):
        T = self.X0.shape[0]
        if self.keep_hid is True and self.H is not None:
            h_temp = self.H[-1,::]
            self.H = np.zeros((T+1, self.n_hid))
            self.H[0, ::] = h_temp
        else:
            self.H = np.zeros((T+1, self.n_hid))
        self.Z_h = np.zeros((T, self.n_hid))
        self.Z_o = np.zeros((T, self.n_out))
        if aux is False:
            for iTm in range(T):
                self.Z_h[iTm, ::] = np.dot(self.X0[iTm, ::], self.params['W_vh']) + np.dot(self.H[iTm, ::], self.params['W_hh']) + self.params['b_h']
                self.H[iTm + 1, ::] = self.g_o(self.Z_h[iTm, ::])
                self.Z_o[iTm, ::] = np.dot(self.H[iTm + 1, ::], self.params['W_ho']) + self.params['b_o']
        else:
            for iTm in range(T):
                self.Z_h[iTm, ::] = np.dot(self.X0[iTm, ::], self.params_aux['W_vh']) + np.dot(self.H[iTm, ::], self.params_aux['W_hh']) + self.params_aux['b_h']
                self.H[iTm + 1, ::] = self.g_o(self.Z_h[iTm, ::])
                self.Z_o[iTm, ::] = np.dot(self.H[iTm + 1, ::], self.params_aux['W_ho']) + self.params_aux['b_o']
                
        self.X1 = self.g_o(self.Z_o)
    
    def backward(self, aux=False):
        self.DH = np.zeros_like(self.H)
        self.D0 = np.zeros_like(self.X0)
        if self.D1 is None:
            G_o = self.g_o_prime(self.Z_o)
        else:
            G_o = np.multiply(self.D1, self.g_o_prime(self.Z_o))

        T = G_o.shape[0]
        G_h = np.zeros_like(self.Z_h)
        if aux is False:
            for iTm in range(T-1, -1, -1):
                G_h[iTm, ::] = np.multiply(self.DH[iTm+1, ::], self.g_h_prime(self.Z_h[iTm, ::]))
                self.DH[iTm , ::] = np.dot(G_o[iTm, ::], self.params['W_ho'].transpose()) + np.dot(G_h[iTm, ::], self.params['W_hh'].transpose())
                self.D0[iTm, ::] = np.dot(G_h[iTm, ::], self.params['W_vh'].transpose())
        else:
            for iTm in range(T-1, -1, -1):
                G_h[iTm, ::] = np.multiply(self.DH[iTm+1, ::], self.g_h_prime(self.Z_h[iTm, ::]))
                self.DH[iTm , ::] = np.dot(G_o[iTm, ::], self.params_aux['W_ho'].transpose()) + np.dot(G_h[iTm, ::], self.params_aux['W_hh'].transpose())
                self.D0[iTm, ::] = np.dot(G_h[iTm, ::], self.params_aux['W_vh'].transpose())
        self.grads['W_ho'] = np.dot(self.H[1:, ::].transpose(), G_o)
        self.grads['b_o'] = np.sum(G_o, axis=0)
        self.grads['W_hh'] = np.dot(self.H[:T, ::].transpose(), G_h)
        self.grads['W_vh'] = np.dot(self.X0.transpose(), G_h)
        self.grads['b_h'] = np.sum(G_h, axis=0)
            
    def updateParam(self, solver_func):
        solver_func(self)
        self.params['W_ho'] += self.deltas['W_ho']
        self.params['b_o'] += self.deltas['b_o']
        self.params['W_hh'] += self.deltas['W_hh']
        self.params['W_vh'] += self.deltas['W_vh']
        self.params['b_h'] += self.deltas['b_h']


class Net(object):
    """
    Net is a container for all the Layer objects that form a network

    It manages the forward,backward passes through the networks as well
    as the parametr update calls. Note that Net objects are independent of
    the cost function employed for their training
    """
    
    n_layer = None
    layers = None
    Xout = None
    n_in = None
    n_out = None
    def __init__(self):
        self.n_layer = 0
        self.layers = []
        self.Xout = None
    
    def addLayer(self, new_layer): # n_in=None, n_out=None, activation=None):
        if self.n_layer > 0:
            assert new_layer.n_in == self.layers[-1].n_out , "New layer does not matche the current net output size"
        self.layers += [new_layer]
        self.n_layer += 1
 
    def forward(self, X, aux=False):
        X0 = X
        for layer in self.layers:
            layer.X0 = X0
            layer.forward(aux=aux)
            X0 = layer.X1
        self.Xout = X0
        
    def backward(self, DeltaN, aux=False):
        Delta1 = DeltaN
        for layer in self.layers[-1::-1]:
            layer.D1 = Delta1
            layer.backward(aux=aux)
            Delta1 = layer.D0

    def updateParam(self, solver_func=None):
        if solver_func is None:
            pass
        else:
            for layer in self.layers[-1::-1]:
                layer.updateParam(solver_func)


class NetTrainer(object):
    """
    NetTrainer iterates over data batches to update Net paramaters that
    minimize a give cost.

    The following are the most important elements are necessary to 
    instantiate a NetTrainer object:
    -- net: Net object to be trained
    -- train_data: given in the form a numpy array with first dimension is the
       number of data exemplars.
    -- label_data: given in asimilar form to train_data. number of exemplars 
       must be consistent with train_data
    -- solver: Solver object that specifies the update rule
    -- loss_func: These function must be chosen from loss_list which is defined        in 'loss_funmctions.py'
    Training parameters are given in the form of a dictionary
    """

    net = None
    batch_size = None
    max_iter = None
    solver_func = None
    loss_func = None
    print_interval = None
    train_data = None
    label_data = None
    shuffle_data = None

    def __init__(self, params):
        for prm_name in params.keys():
            setattr(self, prm_name, params[prm_name])
        if self.max_iter is None:
            self.max_iter = 1000
        assert self.net is not None, "Net object cannot be None"
        assert self.loss_func is not None, "No loss was specified"
        self.loss = loss_list[self.loss_func][0]
        self.lossPrime = loss_list[self.loss_func][1]
        assert self.train_data is not None, "Training data must be specified"
        if self.shuffle_data is not None:
            self.data = Data(self.train_data, batch_size=self.batch_size, shuffle=self.shuffle_data)
        else:
            self.data = Data(self.train_data, batch_size=self.batch_size)
        assert self.label_data is not None, "Labels must be specified"        
        self.labels = Data(self.label_data)
        self.solver_func = self.solver.solver_func
        self.solver.resetAux(self.net)
  
    def train(self, n_iter=None):
        for iTr in range(self.max_iter):
            Xin = self.data.getBatch()
            T = self.labels.getDataAsIn(self.data)
            objective = self.solver.step(self.net, Xin, T, self.loss_func)
            if iTr % self.print_interval == 0:
                print "Iteration %d, objective = %f" % (iTr,objective)

