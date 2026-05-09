'''rnn.py
Recurrent neural networks (RNNs) for text generation
Yazan and Joshua
CS 443: Bio-inspired Machine Learning
Project 4: Recurrent Neural Networks
'''
import numpy as np
import tensorflow as tf

import network
from layers import Dense, Dropout
from skipgram_layers import Embedding
from rnn_layers import GRU

from tf_util import arange_index

class RNN(network.DeepNetwork):
    '''Parent class for all specific types of recurrent neural networks (RNNs).
    '''
    def __init__(self, input_feats_shape, C, pad_token=0, start_token=1, end_token=2):
        '''RNN constructor.

        Parameters:
        -----------
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            For the RNN this is (T, M), where T is the seq len and M is the vocab size.
        C: int.
            Number of classes in the dataset / vocab size.
        pad_token: int.
            Int code for the padding token.
        start_token: int.
            Int code for the start token.
        end_token: int.
            Int code for the end token.

        TODO:
        1. Call the superclass constructor to pass along parameters that `DeepNetwork` has in common.
        2. Create instance variables for parameters as needed.
        '''
        super().__init__(input_feats_shape)
        self.C = C
        self.pad_int = pad_token
        self.start_int = start_token
        self.end_int = end_token

        # KEEP THE FOLLOWING
        # This is a list of boolean values that has length = layers in the net.
        # It holds True if the i-th layer in the network if a recurrent layer, False if not.
        # This is intended to be used to determine during the net's forward pass whether recurrent layers should
        # bootstrap off of an existing state, if it is provided.
        self.is_recurrent_layer = []

    def loss(self, out_net_act, y, mask, eps=1e-8):
        '''Computes the temporal cross entropy loss for the current minibatch based on the output layer activations
        `out_net_act` and int-coded class labels `y`, and padding mask `mask`.

        Parameters:
        -----------
        output_layer_net_act: tf.float32 tensor. shape=(B, T, C).
            Activation in the output layer for all time steps and sequence in the current mini-batch.
        y: tf.int32 tensor. shape=(B, T).
            int-coded next tokens in the vocabulary for every time step in each sequence in the current mini-batch.
        mask: tf.float32 tensor. shape=(B, T, 1).
            Padding mask for time step of each sample in the mini-batch. Values are binary:
            1 if the current token is NOT the padding token.
            0 if the current token IS the padding token.
            We do NOT allow activations at padding chars to count toward the overall loss (since predicting that the
            next char is a padding char is silly!!).
        eps: float.
            Small value to prevent possibly taking the log of 0 or dividing by 0.

        Returns:
        -----------
        float.
            The loss.

        NOTE:
        1. You only need to support the loss type `'temporal_cross_entropy'`. Throw an error if the user is not using
        this loss for the RNN.
        2. Use your regular cross entropy loss code as a starting point for the temporal version.
        3. Note that `arange_index` assumes netAct is 2D and y labels are 1D. Make shape adjustments to make this
        happen...
        4. Use the mask to "gate" or only count loss contributions from sequence time steps where the current token is
        NOT the padding token. Be careful when averaging the loss to normalize by the total non-padding contributions
        ONLY.
        5. If you are using tf.reshape, note that if you going to use @tf.function when training your net
        (likely you are), it likes the desired shape to be wrapped in brackets. For example, tf.reshape(blah, [-1])
        rather than tf.reshape(blah, -1).

        '''
        if self.loss_name == 'temporal_cross_entropy':
            B, T, C = out_net_act.shape
            out_net_act_flat = tf.reshape(out_net_act, [-1, C])
            y_flat = tf.reshape(tf.cast(y, tf.int32), [-1])
            mask_flat = tf.reshape(mask, [-1])

            true_class_probs = arange_index(out_net_act_flat, y_flat)
            true_class_probs = tf.clip_by_value(true_class_probs, eps, 1.0)
            loss = -tf.reduce_sum(mask_flat * tf.math.log(true_class_probs)) / (tf.reduce_sum(mask_flat) + eps)
        else:
            raise ValueError(f'Unknown loss function {self.loss_name}')

        return loss

    def __call__(self, x, mask=None, states=None):
        '''Forward pass through the RNN with the data samples `x`.

        Parameters:
        -----------
        x: tf.float32 tensor. shape=(B, T).
            Mini-batch of input sequences.
        mask: tf.float32 tensor. shape=(B, T, 1) or None.
            Padding mask for time step of each sample in the mini-batch. Values are binary:
            1 if the current token is NOT the padding token.
            0 if the current token IS the padding token.
            We do NOT allow activations at padding chars to count toward the overall loss (since predicting that the
            next char is a padding char is silly!!).
        states: tuple of tf.float32 tensors or None. len=num_recurrent_layers. shape of each entry/tensor: (B, H).
            The latest state in each recurrent layer (e.g. GRU) of the net.
            If None, every recurrent layer should process starting from reset states.

        Returns:
        --------
        tf.float32 tensor. shape=(B, T, C).
            Activation in the output layer for all time steps and sequence in the current mini-batch.
        tuple of tf.float32 tensors. len=num_recurrent_layers. shape of each entry/tensor: (B, H).
            List of final states of each recurrent layer at the end of processing the current mini-batch, converted to
            a tuple.

        NOTE:
        1. Clearly, only recurrent layers should be supplied with the mask and prior states...
        2. Make use of the self.is_recurrent_layer list. It will tell you if the current layer is a recurrent layer or
        not.
        '''
        # Should just be hit when doing pilot forward pass. Otherwise, should always be created in train_step/test_step
        # By default, we don't mask out any states
        if mask is None:
            mask = tf.ones([tf.shape(x)[0], tf.shape(x)[1], 1], dtype=tf.float32)

        rec_layer_states = []
        state_ind = 0
        layer_stack = []
        layer = self.output_layer
        while layer is not None:
            layer_stack.append(layer)
            layer = layer.get_prev_layer_or_block()

        net_act = x
        for i, layer in enumerate(reversed(layer_stack)):
            if self.is_recurrent_layer[i]:
                init_state = None
                if states is not None:
                    init_state = states[state_ind]
                net_act = layer(net_act, mask, state=init_state)
                rec_layer_states.append(net_act[:, -1, :])
                state_ind += 1
            else:
                net_act = layer(net_act)

        return net_act, tuple(rec_layer_states)

    @tf.function
    def train_step(self, x_batch, y_batch):
        '''Completely process a single mini-batch of data during training. This includes:
        1. Performing a forward pass of the data through the entire network.
        2. Computing the loss.
        3. Updating the network parameters using backprop (via update_params method).

        Parameters:
        -----------
        x_batch: tf.float32 tensor. shape=(B, ...).
            A single mini-batch of data packaged up by the fit method.
        y_batch: tf.ints32 tensor. shape=(B,).
            int-coded labels of samples in the mini-batch.

        Returns:
        --------
        float.
            The loss.
        '''
        # Make mask for padding char: 1 if NOT the padding char, 0 if it IS the padding char
        # mask shape: (B, T) -> (B, T, 1) for compatibility with (B, T, H) in rec layers
        mask = tf.expand_dims(tf.cast(x_batch != self.pad_int, dtype=tf.float32), axis=-1)

        # Do forward pass with gradients tracked in the tape
        with tf.GradientTape() as tape:
            net_act, _ = self(x=x_batch, mask=mask)
            loss = self.loss(net_act, y_batch, mask)

        # Do wt update
        self.update_params(tape, loss)
        return loss

    @tf.function
    def test_step(self, x_batch, y_batch):
        '''Completely process a single mini-batch of data during test/validation time. This includes:
        1. Performing a forward pass of the data through the entire network.
        2. Computing the loss.
        3. Obtaining the predicted classes for the mini-batch samples.
        4. Compute the accuracy of the predictions.

        Parameters:
        -----------
        x_batch: tf.float32 tensor. shape=(B, ...).
            A single mini-batch of data packaged up by the fit method.
        y_batch: tf.ints32 tensor. shape=(B,).
            int-coded labels of samples in the mini-batch.

        Returns:
        --------
        float.
            The accuracy.
        float.
            The loss.
        '''
        # Make mask for padding char: 1 if NOT the padding char, 0 if it IS the padding char
        # mask shape: (B, T) -> (B, T, 1) for compatibility with (B, T, H)
        mask = tf.expand_dims(tf.cast(x_batch != self.pad_int, dtype=tf.float32), axis=-1)

        # Validation loss
        # compute validation net_act
        out_net_act_val, _ = self(x=x_batch, mask=mask)
        # compute validation loss
        loss = self.loss(out_net_act_val, y_batch, mask)
        acc = tf.constant(0.0)
        return acc, loss

    def generate(self, prompt, length, char2ind_map, ind2char_map, r_seed=0):
        '''Generates/predicts a sequence of chars of length `length` chars that follow the provided prompt.
        It is helpful remember that the RNN generates chars one at a time sequentially. Therefore in
        prediction/generation mode, the network processes tokens in mini-batches of one item for one time step.

        Parameters:
        -----------
        prompt: str.
            Chars to pass thru the RNN one-at-a-time sequentially to build up the state before the net predicts the
            next char.
        length: int.
            Maximum number of chars that RNN generates after the prompt chars.
            NOTE: The RNN can decide to terminate the text generation early itself if it predicts the <END> token.
        char2ind_map: Python dictionary.
            Keys: chars in vocab. Values: int code of a char in the vocab.
        ind2char_map: Python dictionary.
            Keys: int code of a char in the vocab. Values: Which char it corresponds to in the vocab.
        r_seed: int.
            Random seed to control the randomness a NumPy RNG object that controls the sampling of which token to
            predict next based on the RNN's output layer softmax probabilities.

        Returns:
        --------
        str. len=(len(prompt) + length).
            The provided prompt concatenated with the set of RNN generated chars.

        TODO: Fill in code snippets in the places below marked with a TODO item.
        '''
        # RNG object to control which next token is probablistically selected based on the softmax probs
        rng = np.random.default_rng(r_seed)

        '''1: Convert prompt from str to int'''
        # We want to convert prompt (str) to B, T, M (where B = 1)
        # Convert chars -> int
        prompt_int = [char2ind_map[char] for char in prompt]  # (N_gen,)

        '''2: Warm up RNN state with the prompt'''
        # Warm up RNN to develop GRU state
        # a. Handle 1st token, which we assume will always be the start token
        _, states = self(tf.reshape(tf.constant(self.start_int, dtype=tf.int32), [1, 1]))
        # b. Process the rest of the prompt, except last prompt token. Allow states to progressively build.
        for prompt_token in prompt_int[:-1]:
            x_int_tf = tf.reshape(tf.constant(prompt_token, dtype=tf.int32), [1, 1])
            _, states = self(x_int_tf, states=states)

        '''3: Generate new chars using a feedback loop (prev pred = next input), starting with last prompt char'''
        seq_gen_int = [prompt_int[-1]]
        for t in range(length):
            curr_token = seq_gen_int[-1]

            # Convert int into Tensor format (1, 1)
            x_int_tf = tf.reshape(tf.constant(curr_token, dtype=tf.int32), [1, 1]) # B, T

            net_act, states = self(x_int_tf, states=states)

            out_probs_np = tf.squeeze(net_act, axis=[0, 1]).numpy()
            out_probs_np = out_probs_np / out_probs_np.sum()

            # Draw predicted char index from vocab proportional to the softmax prob
            pred_char_int = rng.choice(np.arange(len(out_probs_np)), p=out_probs_np)
            # Release int from numpy
            pred_char_int = pred_char_int.item()

            if pred_char_int == self.end_int:
                break

            seq_gen_int.append(pred_char_int)

        '''4: Convert the generated int tokens to chars'''
        generated_chars = [ind2char_map[token] for token in seq_gen_int[1:]]

        '''5: Concat the prompt and the generated seq'''
        generated_text = prompt + ''.join(generated_chars)

        return generated_text


class GRU_RNN1Mini(RNN):
    '''Mini recurrent neural network with a single GRU layer.

    Embedding → GRU → Dense

    Both the input and output layer have `vocab_sz` units. The output layer uses regular softmax activation.

    All layers use He/Kaiming weight initialization.
    '''
    def __init__(self, input_feats_shape, C, embedding_dim=64, rnn_units=128):
        '''GRU_RNN1Mini constructor

        Parameters:
        -----------
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            For the RNN this is (T, M), where T is the seq len and M is the vocab size.
        C: int.
            Number of classes in the dataset / vocab size.
        embedding_dim: int.
            Number of neurons in the Embedding layer (M).
        rnn_units: int.
            Number of neurons in the GRU layer (H).

        TODO:
        1. Call the superclass constructor to pass along parameters that `DeepNetwork` has in common.
        2. Build out the network like usual. NOTE: you should populate the self.is_recurrent_layer list.
        '''
        super().__init__(input_feats_shape, C)
        embedding = Embedding(name='Embedding', units=embedding_dim)
        gru = GRU(name='GRU_1', units=rnn_units, prev_layer_or_block=embedding)
        output = Dense(name='Output', units=C, activation='softmax', prev_layer_or_block=gru)
        self.output_layer = output
        self.is_recurrent_layer = [False, True, False]


class GRU_RNN1(GRU_RNN1Mini):
    '''Recurrent neural network with a single GRU layer.

    Embedding → GRU → Dense

    Both the input and output layer have `vocab_sz` units. The output layer uses regular softmax activation.

    All layers use He/Kaiming weight initialization.
    '''
    def __init__(self, input_feats_shape, C, embedding_dim=64, rnn_units=256):
        '''GRU_RNN1 constructor

        Parameters:
        -----------
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            For the RNN this is (T, M), where T is the seq len and M is the vocab size.
        C: int.
            Number of classes in the dataset / vocab size.
        embedding_dim: int.
            Number of neurons in the Embedding layer (M).
        rnn_units: int.
            Number of neurons in the GRU layer (H).

        NOTE: This has the same architecture as GRU_RNN1Mini (only number of units different) so you can build this
        with one line of code :)
        '''
        super().__init__(input_feats_shape, C, embedding_dim=embedding_dim, rnn_units=rnn_units)


class GRU_RNN2(RNN):
    '''Recurrent neural network with a two GRU layers.

    Embedding → GRU → Dropout → GRU → Dropout → Dense

    Both the input and output layer have `vocab_sz` units. The output layer uses regular softmax activation.

    All layers use He/Kaiming weight initialization.
    '''
    def __init__(self, input_feats_shape, C, embedding_dim=64, rnn_units=(384, 384), dropout_rates=(0.0, 0.2)):
        '''GRU_RNN2 constructor

        Parameters:
        -----------
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            For the RNN this is (T, M), where T is the seq len and M is the vocab size.
        C: int.
            Number of classes in the dataset / vocab size.
        embedding_dim: int.
            Number of neurons in the Embedding layer (M).
        rnn_units: tuple of ints.
            Number of neurons in each GRU layer (H).
        dropout_rate: tuple of float
            Dropout rate to use each the Dropout layer in the net.

        TODO:
        1. Call the superclass constructor to pass along parameters that `DeepNetwork` has in common.
        2. Build out the network like usual. NOTE: you should populate the self.is_recurrent_layer list.
        '''
        super().__init__(input_feats_shape, C)
        embedding = Embedding(name='Embedding', units=embedding_dim)
        gru_1 = GRU(name='GRU_1', units=rnn_units[0], prev_layer_or_block=embedding)
        dropout_1 = Dropout(name='Dropout_1', rate=dropout_rates[0], prev_layer_or_block=gru_1)
        gru_2 = GRU(name='GRU_2', units=rnn_units[1], prev_layer_or_block=dropout_1)
        dropout_2 = Dropout(name='Dropout_2', rate=dropout_rates[1], prev_layer_or_block=gru_2)
        output = Dense(name='Output', units=C, activation='softmax', prev_layer_or_block=dropout_2)
        self.output_layer = output
        self.is_recurrent_layer = [False, True, False, True, False, False]


class GRU_RNN2XL(GRU_RNN2):
    '''Larger recurrent neural network with a two GRU layers.

    Embedding → GRU → Dropout → GRU → Dropout → Dense

    Both the input and output layer have `vocab_sz` units. The output layer uses regular softmax activation.

    All layers use He/Kaiming weight initialization.
    '''
    def __init__(self, input_feats_shape, C, embedding_dim=96, rnn_units=[512, 512], dropout_rates=(0.1, 0.2)):
        '''GRU_RNN2 constructor

        Parameters:
        -----------
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            For the RNN this is (T, M), where T is the seq len and M is the vocab size.
        C: int.
            Number of classes in the dataset / vocab size.
        embedding_dim: int.
            Number of neurons in the Embedding layer (M).
        rnn_units: tuple of ints.
            Number of neurons in each GRU layer (H).
        dropout_rate: tuple of float
            Dropout rate to use each the Dropout layer in the net.

        NOTE: This has the same architecture as GRU_RNN2 (only number of units / parameters different) so you can build
        this with one line of code :)
        '''
        super().__init__(input_feats_shape, C, embedding_dim=embedding_dim, rnn_units=rnn_units,
                         dropout_rates=dropout_rates)


class GRU_RNN3(RNN):
    '''Recurrent neural network with a three GRU layers.

    Embedding → GRU → Dropout → GRU → Dropout → GRU → Dropout → Dense

    Both the input and output layer have `vocab_sz` units. The output layer uses regular softmax activation.

    All layers use He/Kaiming weight initialization.
    '''
    def __init__(self, input_feats_shape, C, embedding_dim=96, rnn_units=(512, 512, 512), dropout_rates=(0.2, 0.2, 0.2)):
        '''GRU_RNN3 constructor

        Parameters:
        -----------
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
            For the RNN this is (T, M), where T is the seq len and M is the vocab size.
        C: int.
            Number of classes in the dataset / vocab size.
        embedding_dim: int.
            Number of neurons in the Embedding layer (M).
        rnn_units: tuple of ints.
            Number of neurons in each GRU layer (H).
        dropout_rate: tuple of float
            Dropout rate to use each the Dropout layer in the net.

        TODO:
        1. Call the superclass constructor to pass along parameters that `DeepNetwork` has in common.
        2. Build out the network like usual. NOTE: you should populate the self.is_recurrent_layer list.
        '''
        super().__init__(input_feats_shape, C)
        embedding = Embedding(name='Embedding', units=embedding_dim)
        gru_1 = GRU(name='GRU_1', units=rnn_units[0], prev_layer_or_block=embedding)
        dropout_1 = Dropout(name='Dropout_1', rate=dropout_rates[0], prev_layer_or_block=gru_1)
        gru_2 = GRU(name='GRU_2', units=rnn_units[1], prev_layer_or_block=dropout_1)
        dropout_2 = Dropout(name='Dropout_2', rate=dropout_rates[1], prev_layer_or_block=gru_2)
        gru_3 = GRU(name='GRU_3', units=rnn_units[2], prev_layer_or_block=dropout_2)
        dropout_3 = Dropout(name='Dropout_3', rate=dropout_rates[2], prev_layer_or_block=gru_3)
        output = Dense(name='Output', units=C, activation='softmax', prev_layer_or_block=dropout_3)
        self.output_layer = output
        self.is_recurrent_layer = [False, True, False, True, False, True, False, False]
