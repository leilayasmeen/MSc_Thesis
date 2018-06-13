import tensorflow as tf
import numpy as np
import math
import time
from pixel_cnn_pp.nn import *

def lrelu(x, rate=0.1):
    # return tf.nn.relu(x)
    return tf.maximum(tf.minimum(x * rate, 0), x)

def fc_lrelu(inputs, num_outputs):
    fc = tf.contrib.layers.fully_connected(inputs, num_outputs,
                                           weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                           activation_fn=tf.identity)
    fc = lrelu(fc)
    return fc

def mlp_discriminator(x, reuse=False):
    with tf.variable_scope('d_net') as vs:
        if reuse:
            vs.reuse_variables()
        x = tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])
        fc1 = fc_lrelu(x, 512)
        fc2 = fc_lrelu(fc1, 512)
        fc3 = tf.contrib.layers.fully_connected(fc2, 1, activation_fn=tf.identity)
        return fc3

class ConvolutionalEncoder(object):
    def __init__(self, X, reg_type, latent_dim, z=None):
        '''
            This is the 6-layer architecture for Convolutional Autoencoder
            mentioned in the original paper:
            Stacked Convolutional Auto-Encoders for Hierarchical Feature Extraction

            Note that only the encoder part is implemented as PixelCNN is taken
            as the decoder.
        '''
        self.x = X
        conv1 = conv2d(X, 64, [4, 4], [2, 2], name='encoder_conv1')
        conv1 = lrelu(conv1)
        conv2 = conv2d(conv1, 128, [4, 4], [2, 2], name='encoder_conv2')
        conv2 = lrelu(conv2)
        conv3 = conv2d(conv2, 256, [4, 4], [2, 2], name='encoder_conv3')
        conv3 = lrelu(conv3)
        conv3 = tf.reshape(conv3, [-1, np.prod(conv3.get_shape().as_list()[1:])])
        fc1 = dense(conv3, 512, name='encoder_fc1')
        fc1 = lrelu(fc1)
        self.mean = dense(fc1, latent_dim, name='encoder_mean')
        self.stddev = tf.nn.sigmoid(dense(fc1, latent_dim, name='encoder_stddev'))
        self.stddev = tf.maximum(self.stddev, 0.01)
        self.pred = self.mean + tf.multiply(self.stddev,
                                       tf.random_normal(tf.stack([tf.shape(X)[0], latent_dim])))

        if "elbo" in reg_type:
            self.reg_loss = tf.reduce_sum(-tf.log(self.stddev) + 0.5 * tf.square(self.stddev) +
                                           0.5 * tf.square(self.mean) - 0.5)
        elif "2norm" in reg_type:
            self.reg_loss = tf.reduce_sum(0.5 * tf.square(self.pred))
        elif "center" in reg_type:
            self.reg_loss = tf.reduce_sum(-tf.log(self.stddev) + 0.5 * tf.square(self.mean))
        elif "elbo0_1" in reg_type:
            self.reg_loss = 0.1 * tf.reduce_sum(-tf.log(self.stddev) + 0.5 * tf.square(self.stddev) +
                                           0.5 * tf.square(self.mean) - 0.5)
        elif "no_reg" in reg_type:
            self.reg_loss = 0.0 # Add something for stability
        elif "stein" in reg_type:
            stein_grad = tf.stop_gradient(self.tf_stein_gradient(self.pred, 1.0))
            self.reg_loss = -10000.0 * tf.reduce_sum(tf.multiply(self.pred, stein_grad))
        elif "adv" in reg_type:
            true_samples = tf.random_normal(tf.stack([tf.shape(X)[0], latent_dim]))
            self.d = mlp_discriminator(true_samples)
            self.d_ = mlp_discriminator(self.pred, reuse=True)

            epsilon = tf.random_uniform([], 0.0, 1.0)
            x_hat = epsilon * true_samples + (1 - epsilon) * self.pred
            d_hat = mlp_discriminator(x_hat, reuse=True)

            ddx = tf.gradients(d_hat, x_hat)[0]
            ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
            self.d_grad_loss = tf.reduce_mean(tf.square(ddx - 1.0) * 10.0)

            self.d_loss_x = -tf.reduce_mean(self.d)
            self.d_loss_e = tf.reduce_mean(self.d_)
            self.d_loss = self.d_loss_x + self.d_loss_e + self.d_grad_loss

            self.d_vars = [var for var in tf.global_variables() if 'd_net' in var.name]
            self.d_train = tf.train.AdamOptimizer(learning_rate=0.00002, beta1=0.5, beta2=0.9).minimize(self.d_loss,
                                                                                                       var_list=self.d_vars)
            tf.summary.scalar('d_loss_x', self.d_loss_x)
            tf.summary.scalar('d_loss_e', self.d_loss_e)
            self.reg_loss = -tf.reduce_mean(self.d_)
            self.reg_loss *= 100
        elif "moment" in reg_type:
            mean = tf.reduce_mean(self.pred, axis=0, keep_dims=True)
            var = tf.reduce_mean(tf.square(self.pred - mean), axis=0)
            mean_loss = tf.reduce_mean(tf.abs(mean))
            var_loss = tf.reduce_mean(tf.abs(var - 1.0))
            tf.summary.scalar('mean', mean_loss)
            tf.summary.scalar('variance', var_loss)
            self.reg_loss = mean_loss + var_loss
        elif "kernel" in reg_type:
            true_samples = tf.random_normal(tf.stack([200, latent_dim]))
            pred_kernel = self.compute_kernel(self.pred, self.pred)
            sample_kernel = self.compute_kernel(true_samples, true_samples)
            mix_kernel = self.compute_kernel(self.pred, true_samples)
            self.reg_loss = tf.reduce_mean(pred_kernel) + tf.reduce_mean(sample_kernel) - 2 * tf.reduce_mean(mix_kernel)
            self.reg_loss *= 100000.0
        else:
            print("Unknown regularization %s" % str(reg_type))
            exit(0)
        self.elbo_loss = tf.reduce_mean(-tf.log(self.stddev) + 0.5 * tf.square(self.stddev) +
                                        0.5 * tf.square(self.mean) - 0.5)

    def compute_kernel(self, x, y, sigma_sqr=1.0):
        x_size = tf.shape(x)[0]
        y_size = tf.shape(y)[0]
        dim = tf.shape(x)[1]
        tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
        tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
        kernel = tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / 2.0 / sigma_sqr)
        return kernel

    # x_sample is input of size (batch_size, dim)
    def tf_stein_gradient(seff, x_sample, sigma_sqr):
        x_size = x_sample.get_shape()[0].value
        x_dim = x_sample.get_shape()[1].value
        x_sample = tf.reshape(x_sample, [x_size, 1, x_dim])
        sample_mat_y = tf.tile(x_sample, (1, x_size, 1))
        sample_mat_x = tf.transpose(sample_mat_y, perm=(1, 0, 2))
        kernel_matrix = tf.exp(-tf.reduce_sum(tf.square(sample_mat_x - sample_mat_y), axis=2) / (2 * sigma_sqr * x_dim))
        # np.multiply(-self.kernel(x, y), np.divide(x - y, self.sigma_sqr))./
        tiled_kernel = tf.tile(tf.reshape(kernel_matrix, [x_size, x_size, 1]), [1, 1, x_dim])
        kernel_grad_matrix = tf.multiply(tiled_kernel, tf.div(sample_mat_y - sample_mat_x, sigma_sqr * x_dim))
        gradient = tf.reshape(-x_sample, [x_size, 1, x_dim])  # Gradient of standard Gaussian
        tiled_gradient = tf.tile(gradient, [1, x_size, 1])
        weighted_gradient = tf.multiply(tiled_kernel, tiled_gradient)
        return tf.div(tf.reduce_sum(weighted_gradient, axis=0) +
                      tf.reduce_sum(kernel_grad_matrix, axis=1), x_size)
"""
class ComputeLL:
    def __init__(self, latent_dim):
        self.mean = tf.placeholder(tf.float32, shape=(None, latent_dim))
        self.stddev = tf.placeholder(tf.float32, shape=(None, latent_dim))
        self.sample = tf.placeholder(tf.float32, shape=(None, latent_dim))
        mu = tf.reshape(self.mean, shape=tf.pack([tf.shape(self.mean)[0], 1, latent_dim]))
        mu = tf.tile(mu, tf.pack([1, tf.shape(self.sample)[0], 1]))
        sig = tf.reshape(self.stddev, shape=tf.pack([tf.shape(self.stddev)[0], 1, latent_dim]))
        sig = tf.tile(sig, tf.pack([1, tf.shape(self.sample)[0], 1]))
        z = tf.reshape(self.sample, shape=tf.pack([1, tf.shape(self.sample)[0], latent_dim]))
        z = tf.tile(z, tf.pack([tf.shape(self.mean)[0], 1, 1]))

        coeff = tf.div(1.0 / math.sqrt(2 * math.pi), sig)
        ll = coeff * tf.exp(-tf.div(tf.square(z - mu), 2 * tf.square(sig)))
        ll = tf.reduce_prod(ll, axis=2)
        self.prob = ll


def compute_mutual_information(data, args, sess, encoder_list, ll_compute):
    print("Evaluating Mutual Information")
    start_time = time.time()
    num_batch = 1000
    z_batch_cnt = 10  # This must divide num_batch
    dist_batch_cnt = 10
    assert num_batch % z_batch_cnt == 0
    assert num_batch % dist_batch_cnt == 0
    batch_size = args.batch_size * args.nr_gpu

    sample_batches = np.zeros((num_batch*batch_size, args.latent_dim))
    mean_batches = np.zeros((num_batch*batch_size, args.latent_dim))
    stddev_batches = np.zeros((num_batch*batch_size, args.latent_dim))

    for batch in range(num_batch):
        x = data.next(args.batch_size * args.nr_gpu) # manually retrieve exactly init_batch_size examples
        x = np.split(x, args.nr_gpu)
        feed_dict = {encoder_list[i].x: x[i] for i in range(args.nr_gpu)}

        result = sess.run([encoder.pred for encoder in encoder_list] +
                          [encoder.mean for encoder in encoder_list] +
                          [encoder.stddev for encoder in encoder_list], feed_dict=feed_dict)
        sample = np.concatenate(result[0:args.nr_gpu], 0)
        z_mean = np.concatenate(result[args.nr_gpu:args.nr_gpu*2], 0)
        z_stddev = np.concatenate(result[args.nr_gpu*2:], 0)
        sample_batches[batch*batch_size:(batch+1)*batch_size, :] = sample
        mean_batches[batch*batch_size:(batch+1)*batch_size, :] = z_mean
        stddev_batches[batch*batch_size:(batch+1)*batch_size, :] = z_stddev

    z_batch_size = batch_size * z_batch_cnt
    dist_batch_size = batch_size * dist_batch_cnt
    prob_array = np.zeros((num_batch*batch_size, num_batch*batch_size), dtype=np.float)
    for z_ind in range(num_batch // z_batch_cnt):
        for dist_ind in range(num_batch // dist_batch_cnt):
            mean = mean_batches[dist_ind*dist_batch_size:(dist_ind+1)*dist_batch_size, :]
            stddev = stddev_batches[dist_ind*dist_batch_size:(dist_ind+1)*dist_batch_size, :]
            sample = sample_batches[z_ind*z_batch_size:(z_ind+1)*z_batch_size, :]
            probs = sess.run(ll_compute.prob, feed_dict={ll_compute.mean: mean,
                                                         ll_compute.stddev: stddev,
                                                         ll_compute.sample: sample})
            prob_array[dist_ind*dist_batch_size:(dist_ind+1)*dist_batch_size, z_ind*z_batch_size:(z_ind+1)*z_batch_size] = probs
        # print()
    # print(np.sum(prob_array))
    marginal = np.sum(prob_array, axis=0)
    ratio = np.log(np.divide(np.diagonal(prob_array), marginal)) + np.log(num_batch*batch_size)
    mutual_info = np.mean(ratio)
    print("Mutual Information %f, time elapsed %fs" % (mutual_info, time.time() - start_time))
    return mutual_info
"""