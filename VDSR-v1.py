import numpy as np
import tensorflow as tf
import os
#import imageio
import scipy.ndimage
import scipy.misc
#import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"


image_x = 1920/10
image_y = 1080/10

def imread(path, index, scale=1, gray=True, is_train=False):
    if gray:
        image = np.zeros([len(index), image_y, image_x, 1])
        cnt = 0
        for i in (index):
            if is_train:
                img = scipy.misc.imread(path + str(i).zfill(4) + 'x' + str(scale) + '.png', flatten=True,
                                        mode='YCbCr').astype(np.float) / 255.
            else:
                img = scipy.misc.imread(path + str(i).zfill(4) + '.png', flatten=True, mode='YCbCr').astype(
                    np.float) / 255.
            if img.shape[1] < img.shape[0]:
                img = img.T
            if is_train and img.shape[0] >= 540:
                image[cnt, :, :, 0] = preproc(img[:int(image_y / scale), :int(image_x / scale)], scale)
                cnt += 1
            elif is_train == False and img.shape[0] >= 1080:
                image[cnt, :, :, 0] = img[:int(image_y / scale), :int(image_x / scale)]
                cnt += 1

        return image[:cnt, :, :, :]
    else:
        image = np.zeros([len(index), image_y, image_x, 3])
        cnt = 0
        for i in (index):
            if is_train:
                img = scipy.misc.imread(path + str(i).zfill(4) + 'x' + str(scale) + '.png', mode='YCbCr').astype(
                    np.float) / 255.
            else:
                img = scipy.misc.imread(path + str(i).zfill(4) + '.png', mode='YCbCr').astype(np.float) / 255.
            if img.shape[1] < img.shape[0]:
                img1 = img[:, :, 0].T
                img2 = img[:, :, 1].T
                img3 = img[:, :, 2].T
                img = np.stack([img1, img2, img3], axis=2)
            if is_train:
                image[cnt, :, :, :] = preproc(img[:int(image_y / scale), :int(image_x / scale), :], scale)
            else:
                image[cnt, :, :, :] = img[:int(image_y / scale), :int(image_x / scale), :]
            cnt += 1
        return image


def preproc(image, scale=2, gray=True):
    if gray:
        return scipy.ndimage.interpolation.zoom(image, (scale / 1.), prefilter=False)
    else:
        image1 = scipy.ndimage.interpolation.zoom(image[:, :, 0], (scale / 1.), prefilter=False)
        image2 = scipy.ndimage.interpolation.zoom(image[:, :, 1], (scale / 1.), prefilter=False)
        image3 = scipy.ndimage.interpolation.zoom(image[:, :, 2], (scale / 1.), prefilter=False)
        imageA = np.stack([image1, image2, image3], axis=2)
        return imageA

def imsave(image, path):
    return scipy.misc.imsave(path, image)




tf.reset_default_graph()
device = "/device:GPU:0"

gray = True
epoch_size = 1000
batch_size = 5
total_iter = 800 / batch_size

if gray == True:
    c_dim = 1
else:
    c_dim = 3

learning_rate = 1e-3

with tf.device(device):
    X = tf.placeholder(tf.float32, [None, image_y, image_x, c_dim], name='input')
    Y = tf.placeholder(tf.float32, [None, image_y, image_x, c_dim], name='output')

    weights = {
        'w1': tf.get_variable('w1', shape=[3, 3, 1, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'w2': tf.get_variable('w2', shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'w3': tf.get_variable('w3', shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'w4': tf.get_variable('w4', shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'w5': tf.get_variable('w5', shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'w6': tf.get_variable('w6', shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'w7': tf.get_variable('w7', shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'w8': tf.get_variable('w8', shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'w9': tf.get_variable('w9', shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'w10': tf.get_variable('w10', shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'w11': tf.get_variable('w11', shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'w12': tf.get_variable('w12', shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'w13': tf.get_variable('w13', shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'w14': tf.get_variable('w14', shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'w15': tf.get_variable('w15', shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'w16': tf.get_variable('w16', shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'w17': tf.get_variable('w17', shape=[3, 3, 64, 1], initializer=tf.contrib.layers.xavier_initializer_conv2d())
    }
    biases = {
        'b1': tf.get_variable('b1', shape=[64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'b2': tf.get_variable('b2', shape=[64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'b3': tf.get_variable('b3', shape=[64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'b4': tf.get_variable('b4', shape=[64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'b5': tf.get_variable('b5', shape=[64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'b6': tf.get_variable('b6', shape=[64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'b7': tf.get_variable('b7', shape=[64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'b8': tf.get_variable('b8', shape=[64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'b9': tf.get_variable('b9', shape=[64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'b10': tf.get_variable('b10', shape=[64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'b11': tf.get_variable('b11', shape=[64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'b12': tf.get_variable('b12', shape=[64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'b13': tf.get_variable('b13', shape=[64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'b14': tf.get_variable('b14', shape=[64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'b15': tf.get_variable('b15', shape=[64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'b16': tf.get_variable('b16', shape=[64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
        'b17': tf.get_variable('b17', shape=[1], initializer=tf.contrib.layers.xavier_initializer_conv2d())
    }

with tf.device(device):
    conv1 = tf.nn.relu(tf.nn.conv2d(X, weights['w1'], strides=[1, 1, 1, 1], padding='SAME') + biases['b1'])
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights['w2'], strides=[1, 1, 1, 1], padding='SAME') + biases['b2'])
    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, weights['w3'], strides=[1, 1, 1, 1], padding='SAME') + biases['b3'])
    conv4 = tf.nn.relu(tf.nn.conv2d(conv3, weights['w4'], strides=[1, 1, 1, 1], padding='SAME') + biases['b4'])
    conv5 = tf.nn.relu(tf.nn.conv2d(conv4, weights['w5'], strides=[1, 1, 1, 1], padding='SAME') + biases['b5'])
    conv6 = tf.nn.relu(tf.nn.conv2d(conv5, weights['w6'], strides=[1, 1, 1, 1], padding='SAME') + biases['b6'])
    conv7 = tf.nn.relu(tf.nn.conv2d(conv6, weights['w7'], strides=[1, 1, 1, 1], padding='SAME') + biases['b7'])
    conv8 = tf.nn.relu(tf.nn.conv2d(conv7, weights['w8'], strides=[1, 1, 1, 1], padding='SAME') + biases['b8'])
    conv9 = tf.nn.relu(tf.nn.conv2d(conv8, weights['w9'], strides=[1, 1, 1, 1], padding='SAME') + biases['b9'])
    conv10 = tf.nn.relu(tf.nn.conv2d(conv9, weights['w10'], strides=[1, 1, 1, 1], padding='SAME') + biases['b10'])
    conv11 = tf.nn.relu(tf.nn.conv2d(conv10, weights['w11'], strides=[1, 1, 1, 1], padding='SAME') + biases['b11'])
    conv12 = tf.nn.relu(tf.nn.conv2d(conv11, weights['w12'], strides=[1, 1, 1, 1], padding='SAME') + biases['b12'])
    conv13 = tf.nn.relu(tf.nn.conv2d(conv12, weights['w13'], strides=[1, 1, 1, 1], padding='SAME') + biases['b13'])
    conv14 = tf.nn.relu(tf.nn.conv2d(conv13, weights['w14'], strides=[1, 1, 1, 1], padding='SAME') + biases['b14'])
    conv15 = tf.nn.relu(tf.nn.conv2d(conv14, weights['w15'], strides=[1, 1, 1, 1], padding='SAME') + biases['b15'])
    conv16 = tf.nn.relu(tf.nn.conv2d(conv15, weights['w16'], strides=[1, 1, 1, 1], padding='SAME') + biases['b16'])
    conv17 = tf.nn.conv2d(conv2, weights['w17'], strides=[1, 1, 1, 1], padding='SAME') + biases['b17']
    out = tf.add(X, conv17)

    loss = tf.reduce_mean(tf.square(Y - out))
    optm = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.device(device):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


train_path='images/train_bicubic_x2/'
label_path='images/train_HR/'
result_path='results/SRCNN_v1/'

with tf.device(device):
    for epoch in range(epoch_size):
        avr_psnr = 0
        for i in range(total_iter):
            index = np.random.choice(800, batch_size, replace=False) + 1
            train_image = imread(path=train_path, index=index, is_train=True, scale=2)
            label_image = imread(path=label_path, index=index)

            sess.run(optm, feed_dict={X: train_image, Y: label_image})
            tr_loss = sess.run(loss, feed_dict={X: train_image, Y: label_image})
            psnr = 20 * np.log10(1. / np.sqrt(tr_loss))
            avr_psnr += psnr

        print ('epoch: %3d, Avr_PSNR: %4f' % (epoch, avr_psnr / total_iter))
        img = sess.run(conv3, feed_dict={X: train_image})
        for j in range(img.shape[0]):
            imsave(img[j, :, :, 0], result_path + 'srcnn' + str(j).zfill(4) + '.png')
            imsave(train_image[j, :, :, 0], result_path + 'interpol_' + str(j).zfill(4) + '.png')

