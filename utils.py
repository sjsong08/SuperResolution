import cv2
import numpy as np
import tensorflow as tf
import os
import scipy.ndimage
import scipy.misc

def imread(path, gray=False, mode='YCbCr'):
    if gray==True:
        return scipy.misc.imread(path, flatten=True, mode=mode).astype(np.float)/255.
    else:
        return scipy.misc.imread(path, mode=mode).astype(np.float)/255.

def imsave(image, path):
    return scipy.misc.imsave(path, image)
        
    cv2.imwrite(os.path.join(os.getcwd(), path), image*255.)
    
def get_patches(image, image_size=32, stride=14, is_save=False, path='images/patches/'):
    if len(image.shape)==3:
        h, w, c = image.shape
    else:
        h, w = image.shape
    sub_images=[]
    cnt=0
    for x in range(0, h-image_size, stride):
        for y in range(0, w-image_size, stride):
            sub_image = image[x:x+image_size, y:y+image_size]
            sub_images.append(sub_image)
            cnt+=1
            if is_save:
                if not os.path.isdir(os.path.join(os.getcwd(), path)):
                    os.makedirs(os.path.join(os.getcwd(), path))
                cv2.imwrite(os.path.join(os.getcwd(), path)+str(cnt)+'.png', sub_image)
    return np.array(sub_images)



def imgset_read(path, index, gray=True, is_train=False):
    img_set = []
    if gray:
        for i in (index):
            img = scipy.misc.imread(path+str(i)+'.png', flatten=True, mode='YCbCr').astype(np.float)/255.
            img_set.append(img)
            
        return np.array(img_set)
    else:
        for i in (index):
            img = scipy.misc.imread(path+str(i)+'.png', mode='rgb').astype(np.float)/255.
            img_set.append(img)
        return np.array(img_set)
    
def imgresize(image, scale=2.):
    num_sample = image.shape[0]
    if len(image.shape)==4:
        
        images = np.zeros([image.shape[0], int(image.shape[1]*scale), int(image.shape[2]*scale), image.shape[3]])
        for i in range(num_sample):
            images[i,:,:,0] = cv2.resize(image[i,:,:], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return images
    else:
        images = np.zeros([image.shape[0], int(image.shape[1]*scale), int(image.shape[2]*scale)])
        for i in range(num_sample):
            images[i,:,:] = cv2.resize(image[i,:,:], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return images
    
def preproc(image, scale=2, gray=True):
    if gray:
        return scipy.ndimage.interpolation.zoom(image, (scale/1.), prefilter=False)
    else:
        image1 = scipy.ndimage.interpolation.zoom(image[:,:,0], (scale/1.), prefilter=False)
        image2 = scipy.ndimage.interpolation.zoom(image[:,:,1], (scale/1.), prefilter=False)
        image3 = scipy.ndimage.interpolation.zoom(image[:,:,2], (scale/1.), prefilter=False)
        imageA = np.stack([image1, image2, image3], axis=2)
        return imageA
    
def bicubic_upsize(image, scale=2):
    if len(image.shape)==4:
        bicImg=np.zeros([image.shape[0], image.shape[1]*scale, image.shape[2]*scale, image.shape[3]])
        for i in range(image.shape[0]):
            bicImg[i,:,:,:] = scipy.ndimage.interpolation.zoom(image[i,:,:,:], [scale, scale, 1], prefilter=False)
    else:
        bicImg = scipy.ndimage.interpolation.zoom(image, [scale,scale, 1], prefilter=False)
    return bicImg
    
    
def _phase_shift(I, r):
    # Helper function with main phase shift operation
    bsize, a, b, c = I.get_shape().as_list()
    X = tf.reshape(I, (-1, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X, a, 1, name='split1')  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x) for x in X], axis=2)  # bsize, b, a*r, r
    X = tf.split(X, b, 1, name='split2')  # b, [bsize, a*r, r]
    X = tf.concat([tf.squeeze(x) for x in X], axis=2)  # bsize, a*r, b*r
    return tf.reshape(X, (-1, a*r, b*r, 1))

def PS(X, r, color=False):
    # Main OP that you can arbitrarily use in you tensorflow code
    if color:
        Xc = tf.split(X,3,3) #(3, 3, X)
        X = tf.concat([_phase_shift(x, r) for x in Xc], axis=3)
    else:
        X = _phase_shift(X, r)
    return X

def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keep_dims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
    return squash_factor * unit_vector

def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)