import tensorflow as tf
import numpy as np
import tensorlayer as tl
from models.classifier_models import *
from models.generator_models import *
from invert_and_classify import invert
from data import celebA_input, mnist_input
from attacks import latent_space_graph
from set_dataset import *
import json
import os
from skimage import io

sess = tf.Session()

all_config = json.load(open("config.json"))

config = all_config["train"]
# Dataset Configurations
DATASET = "celebA" if config["dataset"].lower() == "celeba" else "mnist"
gen_func = get_generator(DATASET)
cla_func = get_classifier(DATASET)
inp_func = get_input(DATASET)

# General Config
BATCH_SIZE = config["batch_size"]
ADV_STRENGTH = config["adv_train_eps"]
ADV_THRESH = config["adv_thresh"]
ADV_TRAIN = config["adv_train"]
L2_EPS = config["max_admissible_l2"]

# Iteration counts
MAX_ITERS = config["max_iters"]
LOG_ITERS = config["log_iters"]
SAVE_ITERS = config["save_iters"]
ADV_ITERS = config["adv_iters"]
ITERS_PER_ADV = config["iters_per_adv"]
LAM_INIT = config["lambda_init"]
LAM_LR = config["lambda_lr"]

# GENERAL PROPERTIES
global_config = all_config[DATASET]
Z_DIM = global_config["z_dim"]
GEN_PATH = global_config["generator_path"]
SAVE_DIR = global_config["robust_classifier_path"]
NUM_CLASSES = global_config["num_classes"]
DATA_DIR = global_config["data_dir"]
GAN_VARIANCE = global_config["gan_variance"]
GAN_MU = global_config["gan_mu"]
TOTAL_SHAPE = global_config["total_shape"]

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

LOG_STR = "Iteration: %d | Loss: %f | Error: %f | Inv Error: %f | " 
LOG_STR += "Overpowered Adv Success: %d%%"

ARGS = {
            "batch_size": BATCH_SIZE,
            "z_dim": Z_DIM,
            "inc_iters": 500,
            "gan_mu": GAN_MU
        }

softmax_loss = tf.nn.softmax_cross_entropy_with_logits

x = tf.placeholder(tf.float32, (None, Z_DIM))
out, gen_vars = gen_func(x, reuse=False)
# Restore Generator
g_saver = tf.train.Saver(var_list=gen_vars)
g_saver.restore(sess, GEN_PATH)

def find_adversaries(eps, iters, standard_init=None):
    """
        Acts as an "overpowered" adversary for adversarial training purposes.
        * eps (float): the maximum l-inf perturbation (epsilon)
        * iters (int): the number of iterations to perform the latent-space
        optimization
        * standard_init (None or np.array of size (BATCH_SIZE, Z_DIM)): if this
        is not None, perform a standard (non-overpowered attack) using the
        latent vectors given here 
    """
    overpowered = (standard_init is None)
    # Variable Initialization
    with tf.variable_scope("latent_attack", reuse=True):
        z1 = tf.get_variable("z1")
        z2 = tf.get_variable("z2")
        starter_op = tf.assign(z2, z1)
    trainable_zs = [z1, z2]
    if not overpowered:
        z1 = standard_init
        trainable_zs = [z2]
    lam = tf.Variable(LAM_INIT)
    def _inner_loop(t, old_pct_adv):
        res = latent_space_graph(z1, z2, lam, eps, trainable_zs, LAM_LR, \
                gen_func, cla_func, TOTAL_SHAPE, L2_EPS)
        with tf.control_dependencies(res):
            return t+1, res[3]

    def _cond(t, pct_adv):
        time_stop = tf.less(t, tf.constant(iters))
        adv_stop = tf.less(pct_adv, tf.constant(ADV_THRESH))
        return tf.logical_and(time_stop, adv_stop)

    with tf.control_dependencies([starter_op]):
        t_fin, pct_adv_fin = tf.while_loop(_cond, _inner_loop, [tf.constant(0), tf.constant(0.0)])
    with tf.control_dependencies([t_fin, pct_adv_fin]):
        return pct_adv_fin

def remaining_initializer():
    all_vars = []
    if type(gen_vars) == dict:
        all_vars.extend(gen_vars.values())
    else:
        all_vars.extend(gen_vars)
    return tf.variables_initializer(filter(lambda x:x not in all_vars, \
            tf.global_variables()))

def train():
    """
        The main training loop. Parameters should be configured in config.json,
        including adversarial training toggle, etc.
    """
    bs = BATCH_SIZE
    images, labels = inp_func.inputs(False, DATA_DIR, BATCH_SIZE)
    oh_labels = tf.one_hot(labels, NUM_CLASSES)
    logits, preds, cla_vars = cla_func(images, reuse=False)
    oh_preds = tf.one_hot(preds, NUM_CLASSES)

    # Cross Entropy loss
    cla_loss = softmax_loss(logits=logits, labels=oh_labels) 
    cla_acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(labels, tf.int32), \
        tf.cast(preds, tf.int32)), tf.float32))

    # Inversion graph
    z_inv = tf.Variable(GAN_VARIANCE*np.random.randn(BATCH_SIZE, Z_DIM).astype(np.float32)+GAN_MU)
    inv_loop, _ = invert(z_inv, gen_func, \
                    tf.stop_gradient(images), \
                    ARGS, \
                    reuse=True)
    with tf.control_dependencies([inv_loop]):
        inv_images, _ = gen_func(z_inv, reuse=True)
        inv_logits, inv_preds = cla_func(tf.stop_gradient(inv_images), reuse=True)
        oh_inv_preds = tf.one_hot(inv_preds, NUM_CLASSES)
    inv_cla_loss = softmax_loss(logits=inv_logits, labels=oh_labels)
    inv_cla_acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(labels, tf.int32), \
        tf.cast(inv_preds, tf.int32)), tf.float32))

    # Adversarial loss
    with tf.variable_scope("latent_attack", reuse=False):
        z_a = tf.get_variable("z1", shape=(BATCH_SIZE, Z_DIM), \
                initializer=tf.random_normal_initializer(GAN_MU, GAN_VARIANCE))
        z_b = tf.get_variable("z2", shape=(BATCH_SIZE, Z_DIM), \
                initializer=tf.random_normal_initializer(GAN_MU, GAN_VARIANCE))
    with tf.control_dependencies([tf.variables_initializer([z_a, z_b])]):
        padv_fin = find_adversaries(ADV_STRENGTH, ITERS_PER_ADV)
    with tf.control_dependencies([padv_fin]):
        gen_img1, _ = gen_func(z_a, reuse=True)
        gen_img2, _ = gen_func(z_b, reuse=True)
    logits1, preds1 = cla_func(tf.stop_gradient(gen_img1), reuse=True)
    logits2, preds2 = cla_func(tf.stop_gradient(gen_img2), reuse=True)
    # want to maximize softmax cross-entropy distance between logits
    sm_logits1 = tf.nn.softmax(logits1)
    sm_logits2 = tf.nn.softmax(logits2)
    adv_loss = -tf.reduce_sum( \
            sm_logits1*tf.log(sm_logits2) + sm_logits2*tf.log(sm_logits1), axis=1)
    # filter out images too far away
    image_diffs = tf.square(gen_img1 - gen_img2)
    image_diffs = tf.sqrt(tf.reduce_sum(image_diffs, axis=[1,2,3]))/TOTAL_SHAPE
    adv_loss = tf.reduce_mean(tf.where(tf.less(image_diffs, L2_EPS), \
                adv_loss, tf.zeros_like(adv_loss, tf.float32)))
    #total_cla_loss = 0.5*cla_loss + 0.5*inv_cla_loss
    total_cla_loss = cla_loss
    comb_loss = 2.0*adv_loss + total_cla_loss

    # Create optimizers:
    opt = tf.train.AdamOptimizer()
    minner_op = opt.minimize(total_cla_loss, var_list=cla_vars)
    if ADV_TRAIN:
        adv_minner_op = opt.minimize(comb_loss, var_list=cla_vars)
    else:
        adv_minner_op = opt.minimize(total_cla_loss, var_list=cla_vars)
    sess.run(remaining_initializer())
    tf.train.start_queue_runners(sess)
    saver = tf.train.Saver(cla_vars)
    for i in range(MAX_ITERS):
        if i % ADV_ITERS == ADV_ITERS - 1:
            _, loss_, acc_, adv_loss_, gen_img1_, gen_img2_ = sess.run([adv_minner_op, 
                                                cla_loss,
                                                cla_acc,
                                                adv_loss,
                                                gen_img1,
                                                gen_img2])
            for j in range(BATCH_SIZE):
                io.imsave('../adv_samples/{0}_1.jpg'.format(j),gen_img1_[j])
                io.imsave('../adv_samples/{0}_2.jpg'.format(j),gen_img2_[j])
            print("Finished adv iteration with loss %f" % (adv_loss_,))
        else:
            _, loss_, acc_ = sess.run([minner_op, cla_loss, cla_acc])
        if i % LOG_ITERS == LOG_ITERS - 1:
            loss_, acc_, inv_acc_ = sess.run([cla_loss, cla_acc, inv_cla_acc])
            padv_fin_ = sess.run(padv_fin)
            l_ = np.mean(loss_)
            op_adv_ = int(padv_fin_*100)
            print(LOG_STR % (i, l_, acc_, inv_acc_, op_adv_))
        if i % SAVE_ITERS == SAVE_ITERS - 1:
            path = "%s/model.ckpt-%d" % (SAVE_DIR,i)
            saver.save(sess, path)
            print("Saving to %s/model.ckpt-%d" % (SAVE_DIR,i))

if __name__ == "__main__":
    train()
