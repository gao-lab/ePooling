import tensorflow as tf
import keras as K


class GlobalExpectationPooling1D(K.layers.Layer):
    """Global Expect pooling operation for temporal data.
        # Arguments
            data_format: A string,
                one of `channels_last` (default) or `channels_first`.
                The ordering of the dimensions in the inputs.
                `channels_last` corresponds to inputs with shape
                `(batch, steps, features)` while `channels_first`
                corresponds to inputs with shape
                `(batch, features, steps)`.
            mode: int
            m_trainable: A boolean variable,
                if m_trainable == True, the base will be trainable,
                else the base will be a constant
            m_value: A integer,
                the value of the base to calculate the prob
        # Input shape
            `(batch_size, steps, features,)`
        # Output shape
            2D tensor with shape:
            `(batch_size, features)`
        """

    def __init__(self, mode=0, m_trainable=False, m_value=1, **kwargs):
        super(GlobalExpectationPooling1D, self).__init__(**kwargs)
        self.m_value = m_value
        self.mode = mode
        self.m_trainable = m_trainable

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

    def call(self, x, **kwargs):
        if self.mode == 0:
            # transform the input
            now = tf.transpose(x, [0, 2, 1])
            # x = x - max(x)
            diff_1 = tf.subtract(now, tf.reduce_max(now, axis=-1, keep_dims=True))
            # x = mx
            diff = tf.multiply(diff_1, self.m)
            # prob =  exp(x_i)/sum(exp(x_j))
            prob = tf.nn.softmax(diff)
            # Expectation = sum(Prob*x)
            expectation = tf.reduce_sum(tf.multiply(now, prob), axis=-1, keep_dims=False)
        else:
            # transform the input
            now = tf.transpose(x, [0, 2, 1])
            # x  - mean(x)
            now_diff = tf.subtract(now, tf.reduce_mean(now, axis=-1, keep_dims=True))
            # x = mx
            now_diff_m = tf.multiply(now_diff, self.m)
            # sgn(x)
            sgn_now = tf.sign(now_diff_m)
            # exp(x - mean) * sgn(x - mean(x))  + exp(x - mean(x))
            diff_2 = tf.add(tf.multiply(sgn_now, tf.exp(now_diff_m)), tf.exp(now_diff_m))
            # x = x/2
            diff_now = tf.div(diff_2, 2)
            # Prob = exp(x) / sum(exp(x))
            prob = diff_now / tf.reduce_sum(diff_now, axis=-1, keep_dims=True)
            expectation = tf.reduce_sum(tf.multiply(now, prob), axis=-1, keep_dims=False)
        return expectation

    def get_config(self):
        base_config = super(GlobalExpectationPooling1D, self).get_config()
        return dict(list(base_config.items()))

    def build(self, input_shape):
        if self.m_trainable:
            self.m = self.add_weight(name='m',
                                     shape=(1, 1),
                                     initializer=K.initializers.Constant(value=self.m_value),
                                     trainable=True)
        else:
            self.m = self.add_weight(name='m',
                                     shape=(1, 1),
                                     initializer=K.initializers.Constant(value=self.m_value),
                                     trainable=False)
        super(GlobalExpectationPooling1D, self).build(input_shape)


if __name__ == '__main__':
    pass
