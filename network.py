'''network.py
Deep neural network core functionality implemented with the low-level TensorFlow API.
Yazan and Joshua
CS 443: Bio-Inspired Learning
'''
import time
import os
import numpy as np
import tensorflow as tf

from tf_util import arange_index


class DeepNetwork:
    '''The DeepNetwork class is the parent class for specific networks (e.g. LinearDecoder, NonlinearDecoder).
    '''
    def __init__(self, input_feats_shape):
        self.input_feats_shape = tuple(input_feats_shape)
        self.loss_name = None
        self.output_layer = None
        self.all_net_params = []
        self.opt = None

    def compile(self, loss='cross_entropy', lr=1e-3, print_summary=True):
        self.loss_name = loss
        self.opt = tf.keras.optimizers.Adam(learning_rate=lr)
        x_fake = self.get_one_fake_input()
        _ = self(x_fake)
        self.init_groupnorm_params()
        if print_summary:
            self.summary()
        self.all_net_params = self.get_all_params()

    def get_one_fake_input(self):
        return tf.zeros(shape=(1, *self.input_feats_shape))

    def summary(self):
        print(75*'-')
        layer = self.output_layer
        while layer is not None:
            print(layer)
            layer = layer.get_prev_layer_or_block()
        print(75*'-')

    def set_layer_training_mode(self, is_training):
        layer = self.output_layer
        while layer is not None:
            layer.set_mode(is_training)
            layer = layer.get_prev_layer_or_block()

    def init_groupnorm_params(self):
        layer = self.output_layer
        while layer is not None:
            layer.init_groupnorm_params()
            layer = layer.get_prev_layer_or_block()

    def get_all_params(self, wts_only=False):
        all_net_params = []
        layer = self.output_layer
        while layer is not None:
            if wts_only:
                params = layer.get_wts()
                if params is None:
                    params = []
                if not isinstance(params, list):
                    params = [params]
            else:
                params = layer.get_params()
            all_net_params.extend(params)
            layer = layer.get_prev_layer_or_block()
        return all_net_params

    def accuracy(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)
        correct = tf.equal(y_true, y_pred)
        return tf.reduce_mean(tf.cast(correct, tf.float32))

    def predict(self, x, output_layer_net_act=None):
        if output_layer_net_act is None:
            output_layer_net_act = self(x)
        y_hat = tf.argmax(output_layer_net_act, axis=-1)
        return tf.cast(y_hat, tf.int32)

    def loss(self, out_net_act, y, eps=1e-16):
        if self.loss_name != 'cross_entropy':
            raise ValueError(f"Unsupported loss '{self.loss_name}'. Only 'cross_entropy' is supported for Project 1.")

        y = tf.cast(y, tf.int32)
        p_true = arange_index(out_net_act, y)
        p_true = tf.clip_by_value(p_true, eps, 1.0)
        return -tf.reduce_mean(tf.math.log(p_true))

    def update_params(self, tape, loss):
        grads = tape.gradient(loss, self.all_net_params)
        self.opt.apply_gradients(zip(grads, self.all_net_params))

    def train_step(self, x_batch, y_batch):
        self.set_layer_training_mode(True)
        with tf.GradientTape() as tape:
            out = self(x_batch)
            loss_val = self.loss(out, y_batch)
        self.update_params(tape, loss_val)
        return loss_val

    def test_step(self, x_batch, y_batch):
        self.set_layer_training_mode(False)
        out = self(x_batch)
        loss_val = self.loss(out, y_batch)
        y_pred = self.predict(x_batch, output_layer_net_act=out)
        acc = self.accuracy(y_batch, y_pred)
        return acc, loss_val

    def fit(self, x, y, x_val=None, y_val=None, batch_size=128, max_epochs=10000, val_every=1, print_every=10,
            verbose=True, patience=999, lr_patience=999, lr_decay_factor=0.5, lr_max_decays=12):
        train_loss_hist = []
        val_loss_hist = []
        val_acc_hist = []

        rng = np.random.default_rng(seed=0)
        N = len(x)
        num_batches = int(np.ceil(N / batch_size))
        recent_val_losses = []
        recent_val_losses_lr = []
        num_lr_decays = 0
        self.set_layer_training_mode(True)

        if verbose:
            print("Starting training...")

        e = 0
        for e in range(1, max_epochs + 1):
            t0 = time.time()
            epoch_loss = 0.0
            for _ in range(num_batches):
                batch_inds = rng.integers(low=0, high=N, size=batch_size)
                batch_inds_tf = tf.convert_to_tensor(batch_inds, dtype=tf.int32)
                x_batch = tf.gather(x, batch_inds_tf)
                y_batch = tf.gather(y, batch_inds_tf)
                batch_loss = self.train_step(x_batch, y_batch)
                epoch_loss += float(batch_loss.numpy())
            epoch_loss /= num_batches
            train_loss_hist.append(epoch_loss)

            do_val = (x_val is not None) and (y_val is not None) and (e % val_every == 0)
            if verbose and (e % print_every == 0 or do_val):
                dt = time.time() - t0
                if do_val:
                    val_acc, val_loss = self.evaluate(x_val, y_val)
                    val_acc = float(val_acc.numpy()) if hasattr(val_acc, "numpy") else float(val_acc)
                    val_loss = float(val_loss.numpy()) if hasattr(val_loss, "numpy") else float(val_loss)
                    val_acc_hist.append(val_acc)
                    val_loss_hist.append(val_loss)
                    print(f"Epoch {e}/{max_epochs} | {dt:.2f}s | train_loss={epoch_loss:.6f} "
                          f"| val_loss={val_loss:.6f} | val_acc={val_acc:.4f}")
                    self.set_layer_training_mode(True)

                    if patience < 999:
                        recent_val_losses, stop = self.early_stopping(recent_val_losses, val_loss, patience)
                        if stop:
                            if verbose:
                                print(f"Early stopping at epoch {e} (patience={patience}).")
                            break

                    if lr_patience < 999 and num_lr_decays < lr_max_decays:
                        recent_val_losses_lr, stop_lr = self.early_stopping(recent_val_losses_lr, val_loss, lr_patience)
                        if stop_lr:
                            self.lr_step_decay(lr_decay_factor)
                            num_lr_decays += 1
                            recent_val_losses_lr = []
                else:
                    print(f"Epoch {e}/{max_epochs} | {dt:.2f}s | train_loss={epoch_loss:.6f}")

        if verbose:
            print(f'Finished training after {e} epochs!')
        return train_loss_hist, val_loss_hist, val_acc_hist, e

    def evaluate(self, x, y, batch_sz=64):
        self.set_layer_training_mode(is_training=False)
        N = len(x)
        if batch_sz > N:
            batch_sz = N
        num_batches = N // batch_sz
        if num_batches < 1:
            num_batches = 1

        loss = acc = 0
        for b in range(num_batches):
            curr_x = x[b*batch_sz:(b+1)*batch_sz]
            curr_y = y[b*batch_sz:(b+1)*batch_sz]
            curr_acc, curr_loss = self.test_step(curr_x, curr_y)
            acc += curr_acc
            loss += curr_loss
        acc /= num_batches
        loss /= num_batches
        return acc, loss

    def early_stopping(self, recent_val_losses, curr_val_loss, patience):
        recent_val_losses.append(curr_val_loss)
        max_len = patience + 1
        if len(recent_val_losses) > max_len:
            recent_val_losses.pop(0)

        stop = False
        if len(recent_val_losses) == max_len:
            oldest = recent_val_losses[0]
            more_recent = recent_val_losses[1:]
            if all(oldest <= v for v in more_recent):
                stop = True

        return recent_val_losses, stop

    def lr_step_decay(self, lr_decay_rate):
        old_lr = float(self.opt.learning_rate.numpy())
        print(f"[LR DECAY] learning rate before: {old_lr}")
        new_lr = old_lr * float(lr_decay_rate)
        self.opt.learning_rate = new_lr
        print(f"[LR DECAY] learning rate after:  {float(self.opt.learning_rate.numpy())}")

    def load_wts(self, path='export', filename='params.npz'):
        full_path = os.path.join(path, filename)
        params = dict(np.load(full_path))

        layer = self.output_layer
        while layer is not None:
            if layer.has_wts():
                curr_layer_params = {}
                curr_layer_param_names = [param_name for param_name in params if layer.get_name() + '/' in param_name]
                for curr_param_name in curr_layer_param_names:
                    name_list = curr_param_name.split('/')
                    curr_layer_params[name_list[-1]] = params[curr_param_name]
                layer.load_wts(curr_layer_params)

            layer = layer.get_prev_layer_or_block()

    def save_wts(self, path='export', filename='params.npz'):
        full_path = os.path.join(path, filename)

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        params = {}

        layer = self.output_layer
        while layer is not None:
            if layer.has_wts():
                layer_params = layer.save_wts()
                for param_name in layer_params:
                    params[layer.get_name() + '/' + param_name] = layer_params[param_name]

            layer = layer.get_prev_layer_or_block()
        np.savez_compressed(full_path, **params)
