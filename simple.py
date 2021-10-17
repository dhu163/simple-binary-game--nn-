import numpy as np

class neuron:
    def __init__(inputs):
        self.inputs = inputs
        self.weights = np.random.rand(len(inputs))
        self.output = 0

    @staticmethod
    def RELU(x):
        return x if x>0 else 0

    def compute(inps = None):
        """Optional input fixes input"""
        if inps is None:
            inps = [inp.output for inp in self.inputs]
        s = sum([w*inp for inp, w in zip(inps, self.weights)])
        self.output = RELU(s)

class net:
    def __init__(n_inputs):
        self.n_inputs = n_inputs
        self.layers = []
    def add_layer(n):
        layer = []
        if not len(self.layers):
            last_layer = [0]


            self.output_layer = 



