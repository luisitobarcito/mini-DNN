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
        if self.momentum is None:
            self.momentum = 0
            
    def step(self, net, Xin, T, loss_func):
        loss = loss_list[loss_func][0]
        lossPrime = loss_list[loss_func][1]
        net.forward(Xin)
        objective = np.mean(loss(T, net.Xout), axis=0)
        net.backward(lossPrime(T, net.Xout) / T.shape[0])
        net.updateParam(self.solver_func)
        return objective
        
    def solver_func(self, layer):
        deltas = {}
        for paramname in layer.params.keys():
            layer.deltas[paramname] = self.momentum * layer.deltas[paramname] - self.lr_rate * layer.grads[paramname]


class NAGSolver(SGDSolver):
    """ Nesterov Accelerated gradient 
    """
    def setAux(self, net):
        for layer in net.layers:
            for paramname in layer.params.keys():
                layer.params_aux[paramname] = layer.params[paramname] + self.momentum * layer.deltas[paramname]
            
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


class RMSPropSolver(Solver):
    """ RMS propagation 
        W_aux and b_aux hold the MS gradients
    """
    rms_forget = None
    def __init__(self, params):
        for prm_name in params.keys():
            setattr(self, prm_name, params[prm_name])
        if hasattr(self, 'momentum'):
            print 'Ignoring momentum parameter for RMSPropSolver'
        if self.rms_forget is None:
            self.rms_forget = 0.99

    def resetAux(self, net):
        for layer in net.layers:
            for paramname in layer.params.keys():
                layer.params_aux[paramname] = np.ones_like(layer.params[paramname])
 
            
    def step(self, net, Xin, T, loss_func):
        loss = loss_list[loss_func][0]
        lossPrime = loss_list[loss_func][1]
        net.forward(Xin)        
        objective = np.mean(loss(T, net.Xout), axis=0)
        net.backward(lossPrime(T, net.Xout) / T.shape[0])
        net.updateParam(self.solver_func)
        return objective
        
    def solver_func(self, layer):
        for paramname in layer.params.keys():
            layer.params_aux[paramname] = self.rms_forget * layer.params_aux[paramname] + (1 - self.rms_forget) * (layer.grads[paramname]**2) 
            layer.deltas[paramname] =  - self.lr_rate * layer.grads[paramname] / np.sqrt(layer.params_aux[paramname]) 


class AdaGradSolver(Solver):
    """ AdaDeltaSolver 
        params_aux hold the MS gradients
    """
    def __init__(self, params):
        for prm_name in params.keys():
            setattr(self, prm_name, params[prm_name])

    def resetAux(self, net):
        for layer in net.layers:
            for paramname in layer.params.keys():
                layer.params_aux[paramname] = np.zeros_like(layer.params[paramname])
        
    def step(self, net, Xin, T, loss_func):
        loss = loss_list[loss_func][0]
        lossPrime = loss_list[loss_func][1]
        net.forward(Xin)        
        objective = np.mean(loss(T, net.Xout), axis=0)
        net.backward(lossPrime(T, net.Xout) / T.shape[0])
        net.updateParam(self.solver_func)
        return objective
        
    def solver_func(self, layer):
        for paramname in layer.params.keys():
            layer.params_aux[paramname] = layer.params_aux[paramname] + (layer.grads[paramname]**2)
            layer.deltas[paramname] =  - self.lr_rate * layer.grads[paramname] / np.sqrt(layer.params_aux[paramname]) 


class AdaDeltaSolver(Solver):
    """ AdaDeltaSolver 
        W_aux and b_aux hold the MS gradients
    """
    rms_forget = None
    ada_eps = None
    deltas_aux = None
    grads_aux = None
    def __init__(self, params):
        for prm_name in params.keys():
            setattr(self, prm_name, params[prm_name])
        if hasattr(self, 'lr_rate'):
            print 'Ignoring learning rate for AdaDeltaSolver'
        if self.rms_forget is None:
            self.rms_forget = 0.99
        if self.ada_eps is None:
            self.ada_eps = 1e-10
                
        self.deltas_aux = []
        self.grads_aux = []

    def resetAux(self, net):
        for layer in net.layers:
            deltas_aux = {}
            grads_aux = {}
            for paramname in layer.params.keys():
                deltas_aux[paramname] = np.zeros_like(layer.params[paramname])
                grads_aux[paramname] = np.zeros_like(layer.params[paramname])
                layer.params_aux[paramname] = np.zeros_like(layer.params[paramname])
            self.deltas_aux += [deltas_aux]
            self.grads_aux += [grads_aux]

            
    def updateAux(self, net):
        layer_count = 0
        for layer in net.layers:
            for paramname in layer.params.keys():
                self.grads_aux[layer_count][paramname] = self.rms_forget * self.grads_aux[layer_count][paramname] + (1 - self.rms_forget) * (layer.grads[paramname]**2)
                self.deltas_aux[layer_count][paramname] = self.rms_forget * self.deltas_aux[layer_count][paramname] + (1 - self.rms_forget) * (layer.deltas[paramname]**2) 
                layer.params_aux[paramname] = np.sqrt(self.deltas_aux[layer_count][paramname] + self.ada_eps) / np.sqrt(self.grads_aux[layer_count][paramname] + self.ada_eps)
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
        for paramname in layer.params.keys():
            layer.deltas[paramname] =  -layer.grads[paramname] * layer.params_aux[paramname] 
