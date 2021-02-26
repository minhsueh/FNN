#Reference: "Deep Learning with Functional Inputs", Barinder Thind et. al, 2020

import numpy as np
from keras import backend as K 
from keras.layers import Layer
import tensorflow as tf

class basis():
    def __init__(self, basis_name):
        self.basis_name = basis_name

    def fourier_basis(self, T, M):
        #T: total points for integral path
        #M: basis number
        #output dimension: M*T
        return(np.fft.fft(np.eye(T))[:M,:])




class functional_input_layer(tf.keras.layers.Dense):
    def __init__(self, output_dim, basis, upper_bound, lower_bound, T, **kwargs):  
        
        #basis: the basis for construct functional weight
        #upper_bound: upper bound of integral 
        #lower_bound: lower bound of integral
        #N: # of integrated points between upper_bound and lower_bound
        ''' 
        basis can be declare by basis class
        Example:
        basis_class = basis()
        fourier_basis = basis_class.fourier_basis(N = 3)
        '''
        self.output_dim = output_dim 
        self.basis = basis
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.T = T
        super(functional_input_layer, self).__init__(**kwargs)

       

 

    def build(self, input_shape): 
        self.kernel = self.add_weight(name = 'kernel', 
            shape = (input_shape[1], self.output_dim), 
            initializer = 'normal', trainable = True) 
        super(functional_input_layer, self).build(input_shape)


    def call(self, input_data): 
        #input_data is a dictionary, containing keys: "functional_input", "scalar_input"
        function_input = input_data["functional_input"] #dimension: K*T, K:input number
        scalar_input= input_data["scalar_input"]

        if not isinstance(function_input, np.ndarray):
            try:
                function_input = np.array(function_input)
            else:
                raise TypeError("function_input need to be type numpy.ndarray")

        
        #approximate delta phi for each basis function and for each functional covariate
        function_input_integral = self.basis @ function_input

        self.c = self.add_weight(name = 'kernel', 
            shape = (function_input_integral.shape, self.output_dim), initializer = 'normal', trainable = True) 
        self.w = self.add_weight(
            shape=(len(scalar_input), self.output_dim), initializer="random_normal", trainable=True)
        self.b = self.add_weight(
            shape=(self.output_dim,), initializer="zeros", trainable=True)

        
        return tf.reduce_sum(self.c * function_input_integral) + K.dot(scalar_input, self.w) + self.b

    def compute_output_shape(self, input_shape): 
        return (input_shape[0], self.output_dim)



