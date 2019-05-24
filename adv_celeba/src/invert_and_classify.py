import tensorflow as tf

def invert(proj_z, generator, target_image, args, name="", reuse=False, std=2):
    """
    Input:
        - generator: A python function which takes in a tensor z and
          returns G(z), an image
        - target image: the image for which we want to find the projection: must
          be of the same shape as G(z)
        - args: dictionary containing batch size, latent dimension, and number
          of steps to take
        - std: the expected variance of the latent space (for regularization)
    """
    batch_size = args["batch_size"]
    z_dim = args["z_dim"]
    inc_iters = args["inc_iters"]
    gan_mu = args["gan_mu"]
    with tf.variable_scope("inverter" + name, reuse=reuse):
        opt = tf.train.MomentumOptimizer(100.0, 0.9)
        #opt = tf.train.AdamOptimizer(1.)
        #opt = tf.train.RMSPropOptimizer(10.,momentum=0.9)
    # Optimization Loop
    def _inner_loop(t, last_proj_loss):
        g_out, _ = generator(proj_z, reuse=reuse)
        real_proj_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum( \
                            tf.square(g_out - target_image), axis=[1,2,3])))
        proj_loss = tf.reduce_mean(tf.square(g_out - target_image))
        #mean_reg = tf.square(tf.reduce_mean(proj_z, axis=1))
        #std_reg = tf.square(std - tf.reduce_mean(tf.square(proj_z), axis=1))
        #z_opt = opt.minimize(proj_loss + 0.0001*std_reg + 0.*mean_reg, var_list=[proj_z])
        mean_reg = tf.reduce_mean(tf.square(proj_z-gan_mu), axis=1)
        std_reg = std**2 - tf.reduce_mean(tf.square(proj_z-gan_mu), axis=1)
        z_opt = opt.minimize(proj_loss + 1e-4*std_reg + 1e-4*mean_reg, var_list=[proj_z])
        with tf.control_dependencies([z_opt]):
            return t + 1, real_proj_loss
    # Stopping condition
    def _cond(t, *args):
        return t < inc_iters
    # Run while loop
    loop, proj_loss = tf.while_loop(_cond, _inner_loop, [tf.constant(0), tf.constant(0.0)])
    return loop, proj_loss


def invert_and_classify(im, classifier, generator, args):
    z = invert(generator, im, args)
    im_z = generator(z, reuse=True)
    return classifier(im_z)
