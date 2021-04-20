import tensorflow as tf
import numpy as np
import matplotlib


class WN_Linear(tf.keras.layers.Layer):
    """
    Linear network layer with weight normalization
    """
    def __init__(self, in_dim, out_dim):
        super(WN_Linear, self).__init__()
        w_init = tf.keras.initializers.GlorotUniform()
        self.w = tf.Variable(initial_value=w_init(shape=(in_dim, out_dim), dtype="float32"), trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(out_dim,), dtype="float32"), trainable=True)
        g_init = tf.ones_initializer()
        self.g = tf.Variable(initial_value=g_init(shape=(out_dim,), dtype="float32"), trainable=True)

    def call(self, inputs):
        v = self.w/tf.norm(self.w, axis = 0, keepdims=True)
        out = tf.matmul(inputs, v)
        out = self.g*out + self.b
        return out


class Residual_Block(tf.keras.layers.Layer):
    """
    Network block used in the Residual Network below
    """
    def __init__(self, in_out_out):
        super(Residual_Block, self).__init__()

        _in = in_out_out[0]
        out1 = in_out_out[1]
        out2 = in_out_out[2]

        modules = []
        modules.append(WN_Linear(_in, out1))
        modules.append((tf.keras.layers.Activation(tf.keras.activations.swish)))
        modules.append(WN_Linear(out1, out2))
        self.sequential = tf.keras.Sequential(modules)

    def call(self, inputs):

        out = self.sequential(inputs)

        out += inputs

        return tf.keras.activations.swish(out)




class Residual_Net(tf.keras.Model):
    """
    Residual network with skip connections
    """

    def __init__(self, layer_dims, means, stds, final_act, sigMult):
         super(Residual_Net, self).__init__()
         self.sigMult = sigMult

         self.layer_dims = layer_dims
         self.means, self.stds = means, stds
         self.final_act = final_act
         #self.tanh = tf.keras.layers.Activation(tf.keras.activations.tanh)
         self.swish = tf.keras.layers.Activation(tf.keras.activations.swish)

         in_out_out1 = layer_dims[1:4]
         in_out_out2 = layer_dims[3:6]

         modules = []

         modules.append(WN_Linear(layer_dims[0], layer_dims[1]))
         modules.append(self.swish)
         modules.append(Residual_Block(in_out_out1))
         modules.append(Residual_Block(in_out_out2))
         modules.append(WN_Linear(layer_dims[5], layer_dims[-1]))

         self.sequential = tf.keras.Sequential(modules)

    def call(self, inputs):
        out = (inputs - self.means[:inputs.shape[1]]) / self.stds[:inputs.shape[1]]

        out = self.sequential(out)

        if self.final_act == 'sigmoid':
            return self.sigMult*tf.sigmoid(out)
        elif self.final_act == 'softplus':
            return tf.math.softplus(out)
        elif self.final_act is None:
            return out
        else:
            print('NEITHER')
            return None



def FP_2D(net_p, net_D, net_U, xyt, tape):
    """
    Differential operators that enfore the Fokker Planck equation
    """

    p_out = net_p(xyt)
    D_out = net_D(xyt)
    U_out = net_U(xyt[:, :2])


    p_derivs = tape.gradient(p_out, xyt)

    p_x, p_y, p_t = p_derivs[:, 0:1], p_derivs[:, 1:2], p_derivs[:, 2:3]

    U_derivs = tape.gradient(U_out, xyt)
    U_x, U_y = U_derivs[:, 0:1], U_derivs[:, 1:2]
    fx, fy = -U_x, -U_y

    q1 = tf.math.multiply(fx, p_out)
    q2 = tf.math.multiply(fy, p_out)

    term1 = tape.gradient(q1, xyt)[:, 0:1]
    term2 = tape.gradient(q2, xyt)[:, 1:2]

    Dp = tf.math.multiply(D_out, p_out)
    Dp_x = tape.gradient(Dp, xyt)[:, 0:1]
    Dp_y = tape.gradient(Dp, xyt)[:, 1:2]
    Dp_xx = tape.gradient(Dp_x, xyt)[:, 0:1]
    Dp_yy = tape.gradient(Dp_y, xyt)[:, 1:2]

    residual = -p_t -term1 -term2 + Dp_xx + Dp_yy

    return residual
