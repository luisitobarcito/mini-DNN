""" 
DNN is a little python module to demonstrate 
the basic elements of a deep neural network 
in action.
"""
import numpy as np
import matplotlib.pyplot as plt

from pointwise_activations import func_list
from loss_functions import loss_list


class Solver(object):
    """
    Solver object contains the method employed to update the network
    parameters based on the gradient information.
    """
    lr_rate = None
    rate_decay = None
    
    def __init__(self, params):
        for prm_name in params.keys():
            setattr(self, prm_name, params[prm_name])

    def resetAux(self, net):
        pass
            
    def solverFunc(self):
        pass

    def step(self, net, Xin, loss_func):
        pass        

class SGDSolver(Solver):

    momentum = None
    def __init__(self, params):
        for prm_name in params.keys():
            setattr(self, prm_name, params[prm_name])

    def step(self, net, Xin, T, loss_func):
        loss = loss_list[loss_func][0]
        lossPrime = loss_list[loss_func][1]
        net.forward(Xin)
        objective = np.mean(loss(T, net.Xout), axis=0)
        net.backward(lossPrime(T, net.Xout) / T.shape[0])
        net.updateParam(self.solver_func)
        return objective
        
    def solver_func(self, layer):
        Delta_W = self.momentum * layer.Delta_W - self.lr_rate * np.dot(layer.X0.transpose(), layer.G) 
        Delta_b = self.momentum * layer.Delta_b - self.lr_rate * np.sum(layer.G, axis=0)
        return Delta_W, Delta_b


class NAGSolver(Solver):
    """ Nesterov Accelerated gradient 
    """
    momentum = None
    def __init__(self, params):
        for prm_name in params.keys():
            setattr(self, prm_name, params[prm_name])

    def setAux(self, net):
        for layer in net.layers:
            layer.W_aux = layer.W + self.momentum * layer.Delta_W
            layer.b_aux = layer.b + self.momentum * layer.Delta_b
            
    def step(self, net, Xin, T, loss_func):
        loss = loss_list[loss_func][0]
        lossPrime = loss_list[loss_func][1]
        net.forward(Xin)        
        objective = np.mean(loss(T, net.Xout), axis=0)
        self.setAux(net)
        net.forward(Xin, aux=True)
        net.backward(lossPrime(T, net.Xout) / T.shape[0], aux=True)
        net.updateParam(self.solver_func)
        return objective
        
    def solver_func(self, layer):
        Delta_W = self.momentum * layer.Delta_W - self.lr_rate * np.dot(layer.X0.transpose(), layer.G) 
        Delta_b = self.momentum * layer.Delta_b - self.lr_rate * np.sum(layer.G, axis=0)
        return Delta_W, Delta_b

class RMSPropSolver(Solver):
    """ RMS propagation 
        W_aux and b_aux hold the MS gradients
    """
    rms_forget = 0.99
    def __init__(self, params):
        for prm_name in params.keys():
            setattr(self, prm_name, params[prm_name])
        if hasattr(self, 'momentum'):
            print 'Ignoring momentum parameter for RMSPropSolver'

    def resetAux(self, net):
        for layer in net.layers:
            layer.W_aux = np.ones_like(layer.W)
            layer.b_aux = np.ones_like(layer.b)

            
    def step(self, net, Xin, T, loss_func):
        loss = loss_list[loss_func][0]
        lossPrime = loss_list[loss_func][1]
        net.forward(Xin)        
        objective = np.mean(loss(T, net.Xout), axis=0)
        net.backward(lossPrime(T, net.Xout) / T.shape[0])
        net.updateParam(self.solver_func)
        return objective
        
    def solver_func(self, layer):
        layer.W_aux = self.rms_forget * layer.W_aux + (1 - self.rms_forget) * (np.dot(layer.X0.transpose(), layer.G)**2) 
        layer.b_aux = self.rms_forget * layer.b_aux + (1 - self.rms_forget) * (np.sum(layer.G, axis=0)**2)
        Delta_W =  - self.lr_rate * np.dot(layer.X0.transpose(), layer.G) / np.sqrt(layer.W_aux) 
        Delta_b =  - self.lr_rate * np.sum(layer.G, axis=0) / np.sqrt(layer.b_aux)
        return Delta_W, Delta_b

class AdaGradSolver(Solver):
    """ AdaDeltaSolver 
        W_aux and b_aux hold the MS gradients
    """
    def __init__(self, params):
        for prm_name in params.keys():
            setattr(self, prm_name, params[prm_name])

    def resetAux(self, net):
        for layer in net.layers:
            layer.W_aux = np.zeros_like(layer.W)
            layer.b_aux = np.zeros_like(layer.b)

        
    def step(self, net, Xin, T, loss_func):
        loss = loss_list[loss_func][0]
        lossPrime = loss_list[loss_func][1]
        net.forward(Xin)        
        objective = np.mean(loss(T, net.Xout), axis=0)
        net.backward(lossPrime(T, net.Xout) / T.shape[0])
        net.updateParam(self.solver_func)
        return objective
        
    def solver_func(self, layer):
        layer.W_aux = layer.W_aux + (np.dot(layer.X0.transpose(), layer.G)**2) 
        layer.b_aux = layer.b_aux + (np.sum(layer.G, axis=0)**2)
        Delta_W =  - self.lr_rate * np.dot(layer.X0.transpose(), layer.G) / np.sqrt(layer.W_aux) 
        Delta_b =  - self.lr_rate * np.sum(layer.G, axis=0) / np.sqrt(layer.b_aux)
        return Delta_W, Delta_b


class AdaDeltaSolver(Solver):
    """ AdaDeltaSolver 
        W_aux and b_aux hold the MS gradients
    """
    rms_forget = 0.99
    ada_eps = 1e-10
    Delta_W_aux = None
    Delta_b_aux = None
    G_W_aux = None
    G_b_aux = None
    def __init__(self, params):
        for prm_name in params.keys():
            setattr(self, prm_name, params[prm_name])
        if hasattr(self, 'lr_rate'):
            print 'Ignoring learning rate for AdaDeltaSolver'

    def resetAux(self, net):
        self.Delta_W_aux = []
        self.Delta_b_aux = []
        self.G_W_aux = []
        self.G_b_aux = []
        for layer in net.layers:
            self.Delta_W_aux = self.Delta_W_aux + [np.zeros_like(layer.W)]
            self.Delta_b_aux = self.Delta_b_aux + [np.zeros_like(layer.b)]   
            self.G_W_aux = self.G_W_aux + [np.zeros_like(layer.W)]
            self.G_b_aux = self.G_b_aux + [np.zeros_like(layer.b)]   
            layer.W_aux = np.zeros_like(layer.W)
            layer.b_aux = np.zeros_like(layer.b)   

    def updateAux(self, net):
        layer_count = 0
        for layer in net.layers:
            self.G_W_aux[layer_count] = self.rms_forget * self.G_W_aux[layer_count] + (1 - self.rms_forget) * (np.dot(layer.X0.transpose(), layer.G)**2)
            self.G_b_aux[layer_count] = self.rms_forget * self.G_b_aux[layer_count] + (1 - self.rms_forget) * (np.sum(layer.G, axis=0)**2) 
            self.Delta_W_aux[layer_count] = self.rms_forget * self.Delta_W_aux[layer_count] + (1 - self.rms_forget) * (layer.Delta_W**2) 
            self.Delta_b_aux[layer_count] = self.rms_forget * self.Delta_b_aux[layer_count] + (1 - self.rms_forget) * (layer.Delta_b**2)
            layer.W_aux = np.sqrt(self.Delta_W_aux[layer_count] + self.ada_eps) / np.sqrt(self.G_W_aux[layer_count] + self.ada_eps)
            layer.b_aux = np.sqrt(self.Delta_b_aux[layer_count] + self.ada_eps) / np.sqrt(self.G_b_aux[layer_count] + self.ada_eps)
            layer_count += 1

            
    def step(self, net, Xin, T, loss_func):
        loss = loss_list[loss_func][0]
        lossPrime = loss_list[loss_func][1]
        net.forward(Xin)        
        objective = np.mean(loss(T, net.Xout), axis=0)
        net.backward(lossPrime(T, net.Xout) / T.shape[0])
        self.updateAux(net)
        net.updateParam(self.solver_func)
        return objective
        
    def solver_func(self, layer):
        Delta_W =  -np.dot(layer.X0.transpose(), layer.G) * layer.W_aux 
        Delta_b =  -np.sum(layer.G, axis=0) * layer.b_aux
        return Delta_W, Delta_b
