import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init


def fc(x, scope, nh, *, init_scale=1.0, init_bias=0.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(x, w)+b


def mlp(num_layers=2, num_hidden=64, activation=tf.tanh, layer_norm=False):
    """
    Stack of fully-connected layers to be used in a policy / q-function approximator

    Parameters:
    ----------

    num_layers: int                 number of fully-connected layers (default: 2)

    num_hidden: int                 size of fully-connected layers (default: 64)

    activation:                     activation function (default: tf.tanh)

    Returns:
    -------

    function that builds fully connected network with a given input tensor / placeholder
    """
    def network_fn(X):
        h = tf.layers.flatten(X)
        for i in range(num_layers):
            h = fc(h, 'mlp_fc{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2))
            if layer_norm:
                h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
            h = activation(h)

        return h

    return network_fn


def build_q_func( hiddens=[256], dueling=True, layer_norm=False, **network_kwargs):
    network = mlp(**network_kwargs)
    def q_func_builder(input_placeholder, num_actions, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            latent = network(input_placeholder)
            if isinstance(latent, tuple):
                if latent[1] is not None:
                    raise NotImplementedError("DQN is not compatible with recurrent policies yet")
                latent = latent[0]

            latent = layers.flatten(latent)

            with tf.variable_scope("action_value"):
                action_out = latent
                # print(hiddens)
                for hidden in hiddens:
                    action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                    if layer_norm:
                        action_out = layers.layer_norm(action_out, center=True, scale=True)
                    action_out = tf.nn.relu(action_out)
                action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

            if dueling:
                with tf.variable_scope("state_value"):
                    state_out = latent
                    for hidden in hiddens:
                        state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=None)
                        if layer_norm:
                            state_out = layers.layer_norm(state_out, center=True, scale=True)
                        state_out = tf.nn.relu(state_out)
                    state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
                action_scores_mean = tf.reduce_mean(action_scores, 1)
                action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
                q_out = state_score + action_scores_centered
            else:
                q_out = action_scores
            return q_out

    return q_func_builder
