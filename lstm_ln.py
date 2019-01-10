from __future__ import print_function, division, absolute_import
from keras import backend as K
from keras.layers import LSTM

def sample_normalize(x, _eps=1e-5):
    """centers a set of samples x to have zero mean and unit standard deviation"""
    # keepdims=True the axes which are reduced are left in the result as dimensions with size one
    # axis=-1 means do things across the last axis
    m = K.mean(x, axis=-1, keepdims=True) # could subtract this off earlier
    # std = K.std(x)
    std = K.sqrt(K.var(x, axis=-1, keepdims=True) + _eps) # not using K.std for _eps stability
    return (x-m)/ (std+_eps)


class LSTM_LN(LSTM):
    """
        LSTM with layer normalization
        adds parameters alpha_i (multiplicative), beta_i (additive)
        which are initialized to ones and zeros vectors of same length as the inputs
        see above function
        """
    
    def laynorm(self, x):  # ignore alpha and beta for starters, fix alpha = 1.0, beta = 0.0
        """centers a set of samples x to have zero mean and unit standard deviation"""
        # keepdims=True the axes which are reduced are left in the result as dimensions with size one
        # axis=-1 means do things across the last axis
        m = K.mean(x, axis=-1, keepdims=True) # could subtract this off earlier
        # std = K.std(x)
        std = K.sqrt(K.var(x, axis=-1, keepdims=True) + _eps) # not using K.std for _eps stability
        output = (x-m)/ (std+_eps)
        # output = alpha * output + beta
        # output = alpha * output
        return output
    
    
    def step(self, x, states):
        h_tm1 = states[0]
        c_tm1 = states[1]
        B_U = states[2]
        B_W = states[3]
        
        if self.consume_less == 'gpu':
            # original linear activity
            # z = K.dot(x * B_W[0], self.W) + K.dot(h_tm1 * B_U[0], self.U) + self.b
            # linear activity without bias term self.b # will need to add this back !!! (see what ryankiros does in ln())
            
            ## question: what is B_W[0] for ??? note that it is is may be used in dropout in get_constants() function
            
            # todo: double check that adding self.b here is same as doing LN on each as in ryankiros implementation
            # this may not actually work
            z = self.laynorm(K.dot(x * B_W[0], self.W)) + self.laynorm(K.dot(h_tm1 * B_U[0], self.U)) + self.b
            # seems that ryankiros divides things into inputs from below and recurrent input from before (t-1)
            # and normalizes them
            
            z0 = z[:, :self.output_dim]                         # i_t in Ba2016
            z1 = z[:, self.output_dim: 2 * self.output_dim]     # f_t in Ba2016
            z2 = z[:, 2 * self.output_dim: 3 * self.output_dim] # g_t in Ba2016
            z3 = z[:, 3 * self.output_dim:]                     # o_t in Ba2016
            # normalization
            
            i = self.inner_activation(z0)    # \sigma(i_t)
            f = self.inner_activation(z1)    # \sigma(f_t)
            c = f * c_tm1 + i * self.activation(z2)  # c_t = sigma(f_t) .* c_{t-1} + \sigma(i_t) .* tanh(g_t)
            o = self.inner_activation(z3) # \sigma(o_t) in Ba2016
        
        ## have not applied layer normalization to these code paths yet:: -clm
        else: # CLM: now this does not make sense to me for example x_i is something different depending on 'cpu' vs 'mem'
            if self.consume_less == 'cpu':  # bug? I do not see W ever being applied to the X along the 'cpu' path.
                # maybe the W is applied in preprocess_input? that must be it
                x_i = x[:, :self.output_dim]
                x_f = x[:, self.output_dim: 2 * self.output_dim]
                x_c = x[:, 2 * self.output_dim: 3 * self.output_dim]
                x_o = x[:, 3 * self.output_dim:]
            elif self.consume_less == 'mem':
                x_i = K.dot(x * B_W[0], self.W_i) + self.b_i
                x_f = K.dot(x * B_W[1], self.W_f) + self.b_f
                x_c = K.dot(x * B_W[2], self.W_c) + self.b_c
                x_o = K.dot(x * B_W[3], self.W_o) + self.b_o
            else:
                raise Exception('Unknown `consume_less` mode.')
            
            i = self.inner_activation(x_i + K.dot(h_tm1 * B_U[0], self.U_i))
            f = self.inner_activation(x_f + K.dot(h_tm1 * B_U[1], self.U_f))
            c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1 * B_U[2], self.U_c))
            o = self.inner_activation(x_o + K.dot(h_tm1 * B_U[3], self.U_o))
    
        h = o * self.activation(self.laynorm(c))
        return h, [h, c]

    def preprocess_input(self, x):
        if self.consume_less == 'cpu':
            if 0 < self.dropout_W < 1:
                dropout = self.dropout_W
            else:
                dropout = 0
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[2]
            timesteps = input_shape[1]
            # clm:: Apply y.w + b for every temporal slice y of x.
            x_i = time_distributed_dense(x, self.W_i, self.b_i, dropout, input_dim, self.output_dim, timesteps)
            x_f = time_distributed_dense(x, self.W_f, self.b_f, dropout, input_dim, self.output_dim, timesteps)
            x_c = time_distributed_dense(x, self.W_c, self.b_c, dropout, input_dim, self.output_dim, timesteps)
            x_o = time_distributed_dense(x, self.W_o, self.b_o, dropout,input_dim, self.output_dim, timesteps)
            return K.concatenate([x_i, x_f, x_c, x_o], axis=2)
        else:
            return x
