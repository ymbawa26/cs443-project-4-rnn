'''layers.py
Neural network layers (e.g. Dense, Dropout, etc.) implemented with the low-level TensorFlow API.
Yazan and Joshua
CS 443: Bio-Inspired Learning
'''
import tensorflow as tf


class Layer:
    '''Parent class for all specific neural network layers (e.g. Dense, Dropout). Implements all functionality shared in
    common across different layers (e.g. net_in, net_act).
    '''
    def __init__(self, layer_name, activation, prev_layer_or_block, do_group_norm=False):
        '''Neural network layer constructor. You should not generally make Layers objects, rather you should instantiate
        objects of the subclasses (e.g. Dense, Conv2D).

        Parameters:
        -----------
        layer_name: str.
            Human-readable name for a layer (Dense_0, Dense_1, etc.). Used for debugging and printing summary of net.
        activation: str.
            Name of activation function to apply within the layer (e.g. 'relu', 'linear').
        prev_layer_or_block: Layer (or Layer-like) object.
            Reference to the Layer object that is beneath the current Layer object. `None` if there is no preceding
            layer.
            Example (standard MLP): Input -> Dense_Hidden -> Dense_Output.
                The Dense_Output Layer object has `prev_layer_or_block=Dense_Hidden`.
        do_group_norm. bool:
            Whether to do group normalization in the layer.
            NOTE: Ignore until instructed otherwise later in the semester.

        Stores constructor parameters as instance variables.
        '''
        self.wts = None
        self.b = None
        self.units = None
        self.output_shape = None

        self.is_training = tf.Variable(False, trainable=False)

        self.num_groups = None
        self.gn_gain = None
        self.gn_bias = None
        self.layer_name = layer_name
        self.activation = activation
        self.prev_layer_or_block = prev_layer_or_block
        self.do_group_norm = do_group_norm

        self.tanh_beta = 1.0

    def get_name(self):
        return self.layer_name

    def get_act_fun_name(self):
        return self.activation

    def get_prev_layer_or_block(self):
        return self.prev_layer_or_block

    def get_wts(self):
        return self.wts

    def get_b(self):
        return self.b

    def has_wts(self):
        return False

    def save_wts(self):
        params = {}
        params['wts'] = self.wts.numpy()
        params['b'] = self.b.numpy()
        return params

    def load_wts(self, params):
        self.wts.assign(params['wts'])
        self.b.assign(params['b'])

    def get_num_units(self):
        return self.units

    def set_tanh_beta(self, beta):
        self.tanh_beta = beta

    def set_num_groups(self, groups):
        self.num_groups = groups

    def get_mode(self):
        return bool(self.is_training.numpy())

    def set_mode(self, is_training):
        self.is_training.assign(is_training)

    def init_params(self, input_shape):
        return

    def compute_net_input(self, x):
        return

    def compute_net_activation(self, net_in):
        act = self.activation

        if act == 'linear':
            return tf.identity(net_in)
        if act == 'relu':
            return tf.nn.relu(net_in)
        if act == 'softmax':
            return tf.nn.softmax(net_in, axis=-1)
        raise ValueError(f"Unsupported activation '{act}'. Use 'relu', 'linear', or 'softmax' for Project 1.")

    def __call__(self, x):
        net_in = self.compute_net_input(x)
        net_act = self.compute_net_activation(net_in)

        if self.output_shape is None:
            self.output_shape = list(net_act.shape)

        return net_act

    def get_kaiming_gain(self):
        return 1.0

    def get_params(self):
        params = []

        if self.wts is not None:
            params.append(self.wts)
        if self.b is not None and self.b.trainable:
            params.append(self.b)
        if self.gn_gain is not None:
            params.append(self.gn_gain)
        if self.gn_bias is not None:
            params.append(self.gn_bias)

        return params

    def is_doing_groupnorm(self):
        return bool(getattr(self, "do_group_norm", False))

    def compute_group_norm(self, net_in, eps=0.001):
        return net_in

    def init_groupnorm_params(self):
        if not self.do_group_norm:
            return


class Dense(Layer):
    '''Neural network layer that uses Dense net input.'''
    def __init__(self, name, units, activation='relu', wt_scale=1e-3, prev_layer_or_block=None, wt_init='normal',
                 do_group_norm=False):
        super().__init__(layer_name=name, activation=activation,
                         prev_layer_or_block=prev_layer_or_block, do_group_norm=do_group_norm)
        self.units = int(units)
        self.wt_scale = float(wt_scale)
        self.wt_init = wt_init

    def has_wts(self):
        return True

    def init_params(self, input_shape):
        M = int(input_shape[-1])
        H = int(self.units)

        w_init = tf.random.normal([M, H], mean=0.0, stddev=self.wt_scale, dtype=tf.float32)
        b_init = tf.zeros([H], dtype=tf.float32)

        self.wts = tf.Variable(w_init, trainable=True, name=f"{self.layer_name}_wts")
        self.b = tf.Variable(b_init, trainable=True, name=f"{self.layer_name}_b")

    def compute_net_input(self, x):
        if self.wts is None or self.b is None:
            self.init_params(list(x.shape))

        return tf.linalg.matmul(x, self.wts) + self.b

    def compute_group_norm(self, net_in, eps=0.001):
        return net_in

    def __str__(self):
        return f'Dense layer output({self.layer_name}) shape: {self.output_shape}'


class Dropout(Layer):
    '''A dropout layer that nixes/zeros out a proportion of the net input signals.'''
    def __init__(self, name, rate, prev_layer_or_block=None):
        super().__init__(layer_name=name, activation='linear',
                         prev_layer_or_block=prev_layer_or_block, do_group_norm=False)
        self.rate = float(rate)

    def compute_net_input(self, x):
        if not bool(self.is_training.numpy()):
            return tf.identity(x)

        keep_prob = 1.0 - self.rate
        if keep_prob <= 0.0:
            return tf.zeros_like(x)

        mask = tf.cast(tf.random.uniform(tf.shape(x)) < keep_prob, x.dtype)
        return (x * mask) / keep_prob

    def __str__(self):
        return f'Dropout layer output({self.layer_name}) shape: {self.output_shape}'


class Flatten(Layer):
    '''A flatten layer that flattens the non-batch dimensions of the input signal.'''
    def __init__(self, name, prev_layer_or_block=None):
        super().__init__(layer_name=name, activation='linear',
                         prev_layer_or_block=prev_layer_or_block, do_group_norm=False)

    def compute_net_input(self, x):
        return tf.reshape(x, [tf.shape(x)[0], -1])

    def __str__(self):
        return f'Flatten layer output({self.layer_name}) shape: {self.output_shape}'
