import numpy as np
import h5py
import sys
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#------------------------------Data loading------------------------------
# Parameters
pi = np.pi; exp = np.exp; ln = np.log; sin = np.sin; cos = np.cos; sqrt = np.sqrt; nu = 10**-3
nx = 128; ny = 128; mx = 3*nx//2; my = 3*ny//2; a = 0; b = 2*pi; L = b-a
dx = L/nx; dy = L/ny; T = 3*10**1; dt = 2.5*10**-3; n = int(T/dt); K = 20

trans = np.linspace(a, b, nx+1, dtype=np.float64)
longi = np.linspace(a, b, ny+1, dtype=np.float64)

kx = np.zeros([1,nx//2+1], dtype = np.float64)
ky = np.zeros([ny,1], dtype = np.float64)
for px in range(nx//2+1):
    kx[0,px] = (2*pi/L)*px
for py in range(ny):
    if py < ny/2:
        ky[py,0] = (2*pi/L)*py
    else:
        ky[py,0] = (2*pi/L)*py-ny
k_mag = kx**2+ky**2
k_round = np.around(np.sqrt(k_mag)).astype(int) ; k_max = np.max(k_round)
k_index = np.linspace(0, k_max, k_max+1, dtype = np.int8)


# Modify to user defined paths
Filename = 'user_define'
Training_dir = Filename + 'user_define'
Validation_dir = Filename + 'user_define'
Test_dir = Filename + 'user_define'

# 'nt' represents the number of data time-steps for lead time
## nt = 20 is the case T = 0.5T_L
nt = 20; train_simul = 500; val_simul = 100; test_simul = 50; start = 150; end = start + 100 + nt; use_data = end - start + 1
data_train = np.zeros([train_simul,use_data,ny,nx,1])
for num in range(train_simul):
    ReadTraining = Training_dir + '/Training%d 2D HIT n=%d T=%d dt=%.4lf K=%d data.h5' %(num+1,nx,T,dt,K)
    fr = h5py.File(ReadTraining, 'r')
    data_train[num,:,:,:,:] = fr['w'][start:start+use_data,:,:,:]
    fr.close()

data_val = np.zeros([val_simul,use_data,ny,nx,1])
for num in range(val_simul):
    ReadValidation = Validation_dir + '/Validation%d 2D HIT n=%d T=%d dt=%.4lf K=%d data.h5' %(num+1,nx,T,dt,K)
    fr = h5py.File(ReadValidation, 'r')
    data_val[num,:,:,:,:] = fr['w'][start:start+use_data,:,:,:]
    fr.close()

data_test = np.zeros([test_simul,use_data,ny,nx,1])
for num in range(test_simul):
    ReadTest = Test_dir + '/Test%d 2D HIT n=%d T=%d dt=%.4lf K=%d data.h5' %(num+1,nx,T,dt,K)
    fr = h5py.File(ReadTest, 'r')
    data_test[num,:,:,:,:] = fr['w'][start:start+use_data,:,:,:]
    fr.close()

training_rms = np.std(data_train[:,0,:,:,:], axis = (1,2))
scaling_factor = np.mean(training_rms[:,0])

data_train = data_train/scaling_factor
data_val = data_val/scaling_factor
data_test = data_test/scaling_factor


# Batch sampling
batch_size = 32
batch_x = np.empty([batch_size, ny, nx, 1], dtype = np.float64)
batch_y = np.empty([batch_size, ny, nx, 1], dtype = np.float64)
batch_x_val = np.empty([batch_size, ny, nx, 1], dtype = np.float64)
batch_y_val = np.empty([batch_size, ny, nx, 1], dtype = np.float64)

def get_batch(data_train,int1,int3):
    for b in range(batch_size):
        sim = int1[b]
        t = int3[b]
        batch_x[b,:,:,:] = data_train[sim,t,:,:,:]
        batch_y[b,:,:,:] = data_train[sim,t+nt,:,:,:]
    return batch_x, batch_y

def get_batch_val(data_val,int2,int3):
    for b in range(batch_size):
        sim = int2[b]
        t = int3[b]
        batch_x_val[b,:,:,:] = data_val[sim,t,:,:,:]
        batch_y_val[b,:,:,:] = data_val[sim,t+nt,:,:,:]
    return batch_x_val, batch_y_val

#------------------------------Models------------------------------
def get_weight(name, shape, gain = np.sqrt(2)):
    fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in) # He init
    return tf.get_variable(name, shape = shape, initializer = tf.initializers.random_normal(0, std))

def get_bias(name, shape):
    return tf.get_variable(name, shape = shape, initializer = tf.initializers.zeros())

def conv2d(x, W, b, strides = 1):
    x = tf.concat([x[:,:,-1:],x,x[:,:,:1]], axis = 2)
    x_padded = tf.concat([x[:,-1:,:],x,x[:,:1,:]], axis = 1)
    x = tf.nn.conv2d(x_padded, W, strides = [1, strides, strides, 1], padding = 'VALID')
    x = tf.nn.bias_add(x, b)
    return x

def fully_connect(x, W, b, strides = 1):
    x = tf.nn.conv2d(x, W, strides = [1, strides, strides, 1], padding = 'VALID')
    x = tf.nn.bias_add(x, b)
    return x

def trans_conv2d(x, W, strides = 2):
    dim_x = tf.shape(x); num_batch = dim_x[0:1]
    oshp = [x.shape[1]*2,x.shape[2]*2,x.shape[3]]
    output_shape = tf.concat([num_batch,oshp], axis = 0)
    x = tf.nn.conv2d_transpose(x, W, output_shape = output_shape, strides = [1, strides, strides, 1], padding = 'VALID')
    return x

def act(x, alpha = 0.2):
    with tf.name_scope('LeakyRelu'):
        alpha = tf.constant(alpha, dtype = x.dtype, name = 'alpha')
        return tf.maximum(x * alpha, x)

def PredictionNet(NET_NAME, x, reuse): # x : 128X128
    x_pooled1 = tf.reduce_mean(tf.reshape(x, [-1, ny//32, 32, nx//32, 32, 1]), axis = (2,4)) # x_pooled1 : 4x4
    x_pooled2 = tf.reduce_mean(tf.reshape(x, [-1, ny//16, 16, nx//16, 16, 1]), axis = (2,4)) # x_pooled2 : 8x8
    x_pooled3 = tf.reduce_mean(tf.reshape(x, [-1, ny//8, 8, nx//8, 8, 1]), axis = (2,4)) # x_pooled3 : 16x16
    x_pooled4 = tf.reduce_mean(tf.reshape(x, [-1, ny//4, 4, nx//4, 4, 1]), axis = (2,4)) # x_pooled4 : 32x32
    x_pooled5 = tf.reduce_mean(tf.reshape(x, [-1, ny//2, 2, nx//2, 2, 1]), axis = (2,4)) # x_pooled5 : 64x64

    with tf.variable_scope(NET_NAME, reuse = reuse):
        weights = {
            'wc1' : get_weight('wc1', [3,3,1,64]),
            'wc2' : get_weight('wc2', [3,3,64,64]),
            'wc3' : get_weight('wc3', [3,3,64,64]),
            'wc4' : get_weight('wc4', [3,3,64+1,64]),
            'wc5' : get_weight('wc5', [3,3,64,64]),
            'wc6' : get_weight('wc6', [3,3,64,64]),
            'wc7' : get_weight('wc7', [3,3,64+1,64]),
            'wc8' : get_weight('wc8', [3,3,64,64]),
            'wc9' : get_weight('wc9', [3,3,64,64]),
            'wc10' : get_weight('wc10', [3,3,64+1,64]),
            'wc11' : get_weight('wc11', [3,3,64,64]),
            'wc12' : get_weight('wc12', [3,3,64,64]),
            'wc13' : get_weight('wc13', [3,3,64+1,32]),
            'wc14' : get_weight('wc14', [3,3,32,32]),
            'wc15' : get_weight('wc15', [3,3,32,32]),
            'wc16' : get_weight('wc16', [3,3,32+1,16]),
            'wc17' : get_weight('wc17', [3,3,16,16]),
            'wc18' : get_weight('wc18', [3,3,16,1]),

            'wtc1' : get_weight('wtc1', [2,2,64,64]),
            'wtc2' : get_weight('wtc2', [2,2,64,64]),
            'wtc3' : get_weight('wtc3', [2,2,64,64]),
            'wtc4' : get_weight('wtc4', [2,2,64,64]),
            'wtc5' : get_weight('wtc5', [2,2,32,32]),
        }

        biases = {
            'bc1' : get_bias('bc1', [64]),
            'bc2' : get_bias('bc2', [64]),
            'bc3' : get_bias('bc3', [64]),
            'bc4' : get_bias('bc4', [64]),
            'bc5' : get_bias('bc5', [64]),
            'bc6' : get_bias('bc6', [64]),
            'bc7' : get_bias('bc7', [64]),
            'bc8' : get_bias('bc8', [64]),
            'bc9' : get_bias('bc9', [64]),
            'bc10' : get_bias('bc10', [64]),
            'bc11' : get_bias('bc11', [64]),
            'bc12' : get_bias('bc12', [64]),
            'bc13' : get_bias('bc13', [32]),
            'bc14' : get_bias('bc14', [32]),
            'bc15' : get_bias('bc15', [32]),
            'bc16' : get_bias('bc16', [16]),
            'bc17' : get_bias('bc17', [16]),
            'bc18' : get_bias('bc18', [1]),
        }

        conv1 = conv2d(x_pooled1, weights['wc1'], biases['bc1'])
        conv1 = act(conv1) # conv1 : 4x4x64
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        conv2 = act(conv2) # conv2 : 4x4x64
        conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
        conv3 = act(conv3) # conv3 : 4x4x64
        conv3 = trans_conv2d(conv3, weights['wtc1'])
        conv3 = tf.concat([conv3,x_pooled2], axis=3) # conv3 : 8x8x65

        conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
        conv4 = act(conv4) # conv4 : 8x8x64
        conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])
        conv5 = act(conv5) # conv5 : 8x8x64
        conv6 = conv2d(conv5, weights['wc6'], biases['bc6'])
        conv6 = act(conv6) # conv6 : 8x8x64
        conv6 = trans_conv2d(conv6, weights['wtc2'])
        conv6 = tf.concat([conv6,x_pooled3], axis=3) # conv6 : 16x16x65

        conv7 = conv2d(conv6, weights['wc7'], biases['bc7'])
        conv7 = act(conv7) # conv7 : 16x16x64
        conv8 = conv2d(conv7, weights['wc8'], biases['bc8'])
        conv8 = act(conv8) # conv8 : 16x16x64
        conv9 = conv2d(conv8, weights['wc9'], biases['bc9'])
        conv9 = act(conv9) # conv9 : 16x16x64
        conv9 = trans_conv2d(conv9, weights['wtc3'])
        conv9 = tf.concat([conv9,x_pooled4], axis=3) # conv9 : 32x32x65

        conv10 = conv2d(conv9, weights['wc10'], biases['bc10'])
        conv10 = act(conv10) # conv10 : 32x32x64
        conv11 = conv2d(conv10, weights['wc11'], biases['bc11'])
        conv11 = act(conv11) # conv11 : 32x32x64
        conv12 = conv2d(conv11, weights['wc12'], biases['bc12'])
        conv12 = act(conv12) # conv12 : 32x32x64
        conv12 = trans_conv2d(conv12, weights['wtc4'])
        conv12 = tf.concat([conv12,x_pooled5], axis=3) # conv12 : 64x64x65

        conv13 = conv2d(conv12, weights['wc13'], biases['bc13'])
        conv13 = act(conv13) # conv13 : 64x64x32
        conv14 = conv2d(conv13, weights['wc14'], biases['bc14'])
        conv14 = act(conv14) # conv14 : 64x64x32
        conv15 = conv2d(conv14, weights['wc15'], biases['bc15'])
        conv15 = act(conv15) # conv15 : 64x64x32
        conv15 = trans_conv2d(conv15, weights['wtc5'])
        conv15 = tf.concat([conv15,x], axis=3) # conv15 : 128x128x33

        conv16 = conv2d(conv15, weights['wc16'], biases['bc16'])
        conv16 = act(conv16) # conv16 : 128x128x16
        conv17 = conv2d(conv16, weights['wc17'], biases['bc17'])
        conv17 = act(conv17) # conv17 : 128x128x16
        conv18 = conv2d(conv17, weights['wc18'], biases['bc18'])
        out = conv18 # conv18 : 128x128x1
    return out

def discriminator(NET_NAME, x, reuse): # x : 128X128
    with tf.variable_scope(NET_NAME, reuse = reuse):
        weights = {
            'wc1' : get_weight('wc1', [3,3,2,16]),
            'wc2' : get_weight('wc2', [3,3,16,16]),
            'wc3' : get_weight('wc3', [3,3,16,32]),
            'wc4' : get_weight('wc4', [3,3,32,32]),
            'wc5' : get_weight('wc5', [3,3,32,32]),
            'wc6' : get_weight('wc6', [3,3,32,64]),
            'wc7' : get_weight('wc7', [3,3,64,64]),
            'wc8' : get_weight('wc8', [3,3,64,64]),
            'wc9' : get_weight('wc9', [3,3,64,64]),
            'wc10' : get_weight('wc10', [3,3,64,64]),
            'wc11' : get_weight('wc11', [3,3,64,64]),
            'wc12' : get_weight('wc12', [3,3,64,64]),
            'wc13' : get_weight('wc13', [3,3,64,64]),
            'wc14' : get_weight('wc14', [3,3,64,64]),
            'wc15' : get_weight('wc15', [3,3,64,64]),
            'wc16' : get_weight('wc16', [3,3,64,64]),
            'wc17' : get_weight('wc17', [3,3,64,64]),
            'wc18' : get_weight('wc18', [3,3,64,64]),

            'wfc1' : get_weight('wfc1', [2,2,64,256]),
            'wfc2' : get_weight('wfc2', [1,1,256,1]),
        }

        biases = {
            'bc1' : get_bias('bc1', [16]),
            'bc2' : get_bias('bc2', [16]),
            'bc3' : get_bias('bc3', [32]),
            'bc4' : get_bias('bc4', [32]),
            'bc5' : get_bias('bc5', [32]),
            'bc6' : get_bias('bc6', [64]),
            'bc7' : get_bias('bc7', [64]),
            'bc8' : get_bias('bc8', [64]),
            'bc9' : get_bias('bc9', [64]),
            'bc10' : get_bias('bc10', [64]),
            'bc11' : get_bias('bc11', [64]),
            'bc12' : get_bias('bc12', [64]),
            'bc13' : get_bias('bc13', [64]),
            'bc14' : get_bias('bc14', [64]),
            'bc15' : get_bias('bc15', [64]),
            'bc16' : get_bias('bc16', [64]),
            'bc17' : get_bias('bc17', [64]),
            'bc18' : get_bias('bc18', [64]),

            'bfc1' : get_bias('bfc1', [256]),
            'bfc2' : get_bias('bfc2', [1]),
        }

        conv1 = conv2d(x, weights['wc1'], biases['bc1'])
        conv1 = act(conv1) # conv1 : 128x128x16
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        conv2 = act(conv2) # conv2 : 128x128x16
        conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
        conv3 = act(conv3) # conv3 : 128x128x32
        conv3 = tf.reduce_mean(tf.reshape(conv3, [-1, ny//2, 2, nx//2, 2, 32]), axis = (2,4)) # conv3 : 64x64x32

        conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
        conv4 = act(conv4) # conv4 : 64x64x32
        conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])
        conv5 = act(conv5) # conv5 : 64x64x32
        conv6 = conv2d(conv5, weights['wc6'], biases['bc6'])
        conv6 = act(conv6) # conv6 : 64x64x64
        conv6 = tf.reduce_mean(tf.reshape(conv6, [-1, ny//4, 2, nx//4, 2, 64]), axis = (2,4)) # conv6 : 32x32x64

        conv7 = conv2d(conv6, weights['wc7'], biases['bc7'])
        conv7 = act(conv7) # conv7 : 32x32x64
        conv8 = conv2d(conv7, weights['wc8'], biases['bc8'])
        conv8 = act(conv8) # conv8 : 32x32x64
        conv9 = conv2d(conv8, weights['wc9'], biases['bc9'])
        conv9 = act(conv9) # conv9 : 32x32x64
        conv9 = tf.reduce_mean(tf.reshape(conv9, [-1, ny//8, 2, nx//8, 2, 64]), axis = (2,4)) # conv9 : 16x16x64

        conv10 = conv2d(conv9, weights['wc10'], biases['bc10'])
        conv10 = act(conv10) # conv10 : 16x16x64
        conv11 = conv2d(conv10, weights['wc11'], biases['bc11'])
        conv11 = act(conv11) # conv11 : 16x16x64
        conv12 = conv2d(conv11, weights['wc12'], biases['bc12'])
        conv12 = act(conv12) # conv12 : 16x16x64
        conv12 = tf.reduce_mean(tf.reshape(conv12, [-1, ny//16, 2, nx//16, 2, 64]), axis = (2,4)) # conv12 : 8x8x64

        conv13 = conv2d(conv12, weights['wc13'], biases['bc13'])
        conv13 = act(conv13) # conv13 : 8x8x64
        conv14 = conv2d(conv13, weights['wc14'], biases['bc14'])
        conv14 = act(conv14) # conv14 : 8x8x64
        conv15 = conv2d(conv14, weights['wc15'], biases['bc15'])
        conv15 = act(conv15) # conv15 : 8x8x64
        conv15 = tf.reduce_mean(tf.reshape(conv15, [-1, ny//32, 2, nx//32, 2, 64]), axis = (2,4)) # conv15 : 4x4x64

        conv16 = conv2d(conv15, weights['wc16'], biases['bc16'])
        conv16 = act(conv16) # conv16 : 4x4x64
        conv17 = conv2d(conv16, weights['wc17'], biases['bc17'])
        conv17 = act(conv17) # conv17 : 4x4x64
        conv18 = conv2d(conv17, weights['wc18'], biases['bc18'])
        conv18 = act(conv18) # conv18 : 4x4x64
        conv18 = tf.reduce_mean(tf.reshape(conv18, [-1, ny//64, 2, nx//64, 2, 64]), axis = (2,4)) # conv18 : 2x2x64

        fc1 = fully_connect(conv18, weights['wfc1'], biases['bfc1'])
        fc1 = act(fc1) # fc1 : 1x1x256
        fc2 = fully_connect(fc1, weights['wfc2'], biases['bfc2'])
        out = tf.reshape(fc2, [-1, 1]) # fc2 : 1x1x1
    return out

def ControlNet(NET_NAME, x, reuse): # x : 128X128
    x_pooled1 = tf.reduce_mean(tf.reshape(x, [-1, ny//32, 32, nx//32, 32, 1]), axis = (2,4)) # x_pooled1 : 4x4
    x_pooled2 = tf.reduce_mean(tf.reshape(x, [-1, ny//16, 16, nx//16, 16, 1]), axis = (2,4)) # x_pooled2 : 8x8
    x_pooled3 = tf.reduce_mean(tf.reshape(x, [-1, ny//8, 8, nx//8, 8, 1]), axis = (2,4)) # x_pooled3 : 16x16
    x_pooled4 = tf.reduce_mean(tf.reshape(x, [-1, ny//4, 4, nx//4, 4, 1]), axis = (2,4)) # x_pooled4 : 32x32
    x_pooled5 = tf.reduce_mean(tf.reshape(x, [-1, ny//2, 2, nx//2, 2, 1]), axis = (2,4)) # x_pooled5 : 64x64

    with tf.variable_scope(NET_NAME, reuse = reuse):
        weights = {
            'wc1' : get_weight('wc1', [3,3,1,64]),
            'wc2' : get_weight('wc2', [3,3,64,64]),
            'wc3' : get_weight('wc3', [3,3,64,64]),
            'wc4' : get_weight('wc4', [3,3,64+1,64]),
            'wc5' : get_weight('wc5', [3,3,64,64]),
            'wc6' : get_weight('wc6', [3,3,64,64]),
            'wc7' : get_weight('wc7', [3,3,64+1,64]),
            'wc8' : get_weight('wc8', [3,3,64,64]),
            'wc9' : get_weight('wc9', [3,3,64,64]),
            'wc10' : get_weight('wc10', [3,3,64+1,64]),
            'wc11' : get_weight('wc11', [3,3,64,64]),
            'wc12' : get_weight('wc12', [3,3,64,64]),
            'wc13' : get_weight('wc13', [3,3,64+1,32]),
            'wc14' : get_weight('wc14', [3,3,32,32]),
            'wc15' : get_weight('wc15', [3,3,32,32]),
            'wc16' : get_weight('wc16', [3,3,32+1,16]),
            'wc17' : get_weight('wc17', [3,3,16,16]),
            'wc18' : get_weight('wc18', [3,3,16,1]),

            'wtc1' : get_weight('wtc1', [2,2,64,64]),
            'wtc2' : get_weight('wtc2', [2,2,64,64]),
            'wtc3' : get_weight('wtc3', [2,2,64,64]),
            'wtc4' : get_weight('wtc4', [2,2,64,64]),
            'wtc5' : get_weight('wtc5', [2,2,32,32]),
        }

        biases = {
            'bc1' : get_bias('bc1', [64]),
            'bc2' : get_bias('bc2', [64]),
            'bc3' : get_bias('bc3', [64]),
            'bc4' : get_bias('bc4', [64]),
            'bc5' : get_bias('bc5', [64]),
            'bc6' : get_bias('bc6', [64]),
            'bc7' : get_bias('bc7', [64]),
            'bc8' : get_bias('bc8', [64]),
            'bc9' : get_bias('bc9', [64]),
            'bc10' : get_bias('bc10', [64]),
            'bc11' : get_bias('bc11', [64]),
            'bc12' : get_bias('bc12', [64]),
            'bc13' : get_bias('bc13', [32]),
            'bc14' : get_bias('bc14', [32]),
            'bc15' : get_bias('bc15', [32]),
            'bc16' : get_bias('bc16', [16]),
            'bc17' : get_bias('bc17', [16]),
            'bc18' : get_bias('bc18', [1]),
        }

        conv1 = conv2d(x_pooled1, weights['wc1'], biases['bc1'])
        conv1 = act(conv1) # conv1 : 4x4
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        conv2 = act(conv2) # conv2 : 4x4
        conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
        conv3 = act(conv3) # conv3 : 4x4
        conv3 = trans_conv2d(conv3, weights['wtc1'])
        conv3 = tf.concat([conv3,x_pooled2], axis=3) # conv3 : 8x8

        conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
        conv4 = act(conv4) # conv4 : 8x8
        conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])
        conv5 = act(conv5) # conv5 : 8x8
        conv6 = conv2d(conv5, weights['wc6'], biases['bc6'])
        conv6 = act(conv6) # conv6 : 8x8
        conv6 = trans_conv2d(conv6, weights['wtc2'])
        conv6 = tf.concat([conv6,x_pooled3], axis=3) # conv6 : 16x16

        conv7 = conv2d(conv6, weights['wc7'], biases['bc7'])
        conv7 = act(conv7) # conv7 : 16x16
        conv8 = conv2d(conv7, weights['wc8'], biases['bc8'])
        conv8 = act(conv8) # conv8 : 16x16
        conv9 = conv2d(conv8, weights['wc9'], biases['bc9'])
        conv9 = act(conv9) # conv9 : 16x16
        conv9 = trans_conv2d(conv9, weights['wtc3'])
        conv9 = tf.concat([conv9,x_pooled4], axis=3) # conv9 : 32x32

        conv10 = conv2d(conv9, weights['wc10'], biases['bc10'])
        conv10 = act(conv10) # conv10 : 32x32
        conv11 = conv2d(conv10, weights['wc11'], biases['bc11'])
        conv11 = act(conv11) # conv11 : 32x32
        conv12 = conv2d(conv11, weights['wc12'], biases['bc12'])
        conv12 = act(conv12) # conv12 : 32x32
        conv12 = trans_conv2d(conv12, weights['wtc4'])
        conv12 = tf.concat([conv12,x_pooled5], axis=3) # conv12 : 64x64

        conv13 = conv2d(conv12, weights['wc13'], biases['bc13'])
        conv13 = act(conv13) # conv13 : 64x64
        conv14 = conv2d(conv13, weights['wc14'], biases['bc14'])
        conv14 = act(conv14) # conv14 : 64x64
        conv15 = conv2d(conv14, weights['wc15'], biases['bc15'])
        conv15 = act(conv15) # conv15 : 64x64
        conv15 = trans_conv2d(conv15, weights['wtc5'])
        conv15 = tf.concat([conv15,x], axis=3) # conv15 : 128x128

        conv16 = conv2d(conv15, weights['wc16'], biases['bc16'])
        conv16 = act(conv16) # conv16 : 128x128
        conv17 = conv2d(conv16, weights['wc17'], biases['bc17'])
        conv17 = act(conv17) # conv17 : 128x128
        conv18 = conv2d(conv17, weights['wc18'], biases['bc18'])
        out = conv18 # conv18 : 128x128
    return out

def spectral_grad(omega):
    s = omega.shape
    omega = tf.transpose(omega[:,:s[1],:s[2],:s[3]], [0, 3, 1, 2]) # [B,C,H,W]
    omegak = tf.signal.rfft2d(omega)/(nx*ny)
    omega_xk = 1J*KX*omegak; omega_yk = 1J*KY*omegak

    omega_x = tf.signal.irfft2d(omega_xk)*(ny*nx); omega_x = tf.transpose(omega_x, [0, 2, 3, 1])
    omega_y = tf.signal.irfft2d(omega_yk)*(ny*nx); omega_y = tf.transpose(omega_y, [0, 2, 3, 1])
    return omega_x, omega_y

#------------------------------Loss functions------------------------------
tf.reset_default_graph()

x = tf.placeholder(tf.float32, [None, ny, nx, 1]) # [B,H,W,C]
y = tf.placeholder(tf.float32, [None, ny, nx, 1])
KX = np.reshape(kx, [1,1,1,nx//2+1]); KX = tf.constant(KX, dtype = tf.complex64)
KY = np.reshape(ky, [1,1,ny,1]); KY = tf.constant(KY, dtype = tf.complex64)

#--------------------conditional GAN (PredictionNet)--------------------
y_pred = PredictionNet('NN1', x, reuse = False)
mean_squared_error = tf.reduce_mean(tf.reduce_mean((y_pred - y)**2, axis = (1,2)))

d1_input_true = tf.concat([y,x], axis = 3)
d1_input_false = tf.concat([y_pred,x], axis = 3)
d1_true = discriminator('D1', d1_input_true, reuse = False)
d1_true_loss = tf.reduce_mean(d1_true)
d1_false = discriminator('D1', d1_input_false, reuse = True)
d1_false_loss = tf.reduce_mean(d1_false)

epsilon1 = tf.random_uniform(shape=[batch_size, 1, 1, 1], minval=0.0, maxval=1.0)
d1_input_hat = d1_input_true + epsilon1 * (d1_input_false - d1_input_true)
d1_hat = discriminator('D1', d1_input_hat, reuse = True)
d1_grad = tf.gradients(d1_hat, d1_input_hat)[0]
slopes1 = tf.sqrt(tf.reduce_sum(tf.square(d1_grad), axis=[1,2,3]))
gp_loss1 = tf.reduce_mean((slopes1 - 1.0)**2)

NN1_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'NN1')
D1_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'D1')
cost_D1 = -d1_true_loss + d1_false_loss + 10*gp_loss1 + 0.001*tf.reduce_mean(d1_true_loss**2)
cost_NN1 = 100*mean_squared_error - d1_false_loss

#--------------------baseline CNN--------------------
y_CNN = PredictionNet('CNN', x, reuse = False)
CNN_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'CNN')
w_loss_CNN = tf.add_n([tf.nn.l2_loss(v) for v in CNN_vars if 'bc' not in v.name])
MSE_CNN = tf.reduce_mean(tf.reduce_mean((y_CNN - y)**2, axis = (1,2)))
cost_CNN = 1*MSE_CNN + 0.0001*w_loss_CNN

#--------------------ControlNet--------------------
x_disturb = ControlNet('NN2', x, reuse = False)
x_disturb = x_disturb - tf.reduce_mean(x_disturb, axis = (1,2), keepdims = True)
x_disturb = 0.1*tf.math.reduce_std(x, axis = (1,2), keepdims = True)\
               *x_disturb/tf.math.reduce_std(x_disturb, axis = (1,2), keepdims = True)
y_disturb_pred = PredictionNet('NN1', x + x_disturb, reuse = True)
disturbed_MSE = tf.reduce_mean(tf.reduce_mean((y_pred - y_disturb_pred)**2, axis = (1,2)))

gradient_x, gradient_y = spectral_grad(x_disturb)
grad_x = tf.reduce_mean(tf.reduce_mean(gradient_x**2, axis = (1,2)))
grad_y = tf.reduce_mean(tf.reduce_mean(gradient_y**2, axis = (1,2)))

NN2_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'NN2')
cost_NN2 = -disturbed_MSE + 0.5*(grad_x + grad_y)

#--------------------Optimizers--------------------
learning_rate = 10**-4; learning_rate2 = 10**-4
trainer_NN1 = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = 0).minimize(cost_NN1, var_list = NN1_vars)
trainer_D1 = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = 0).minimize(cost_D1, var_list = D1_vars)
trainer_CNN = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = 0).minimize(cost_CNN, var_list = CNN_vars)
trainer_NN2 = tf.train.AdamOptimizer(learning_rate = learning_rate2).minimize(cost_NN2, var_list = NN2_vars)

saver1 = tf.train.Saver(var_list = NN1_vars, max_to_keep = None)
NN1_Save_Dir = Filename + 'user_define'
if not os.path.exists(NN1_Save_Dir): os.makedirs(NN1_Save_Dir)

saver_D1 = tf.train.Saver(var_list = D1_vars, max_to_keep = None)
D1_Save_Dir = Filename + 'user_define'
if not os.path.exists(D1_Save_Dir): os.makedirs(D1_Save_Dir)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

#------------------------------Training loop example------------------------------
iterations = 3*10**5
for iter in range(iterations+1):
    int1 = np.random.randint(train_simul, size = batch_size)
    int3 = np.random.randint(use_data-nt, size = batch_size)
    batch_x, batch_y = get_batch(data_train,int1,int3)

    # phase shift
    del_x = 2*pi*np.random.random(); del_y = 2*pi*np.random.random()
    k_dot_delta = np.reshape(del_x*kx + del_y*ky, [1,ny,nx//2+1,1])
    batch_xk = np.fft.rfft2(batch_x, axes = (1,2))/(nx*ny); batch_yk = np.fft.rfft2(batch_y, axes = (1,2))/(nx*ny)
    batch_xk_shift = exp(-1J*k_dot_delta)*batch_xk; batch_yk_shift = exp(-1J*k_dot_delta)*batch_yk
    batch_x = np.fft.irfft2(batch_xk_shift, axes = (1,2))*(nx*ny)
    batch_y = np.fft.irfft2(batch_yk_shift, axes = (1,2))*(nx*ny)

    _,d_loss,d_true,d_false = sess.run([trainer_D1,cost_D1,d1_true_loss,d1_false_loss], feed_dict = {x: batch_x, y: batch_y})
    _,cost_curr,mse_curr = sess.run([trainer_NN1,cost_NN1,mean_squared_error], feed_dict = {x: batch_x, y: batch_y})

    if iter%2000 == 0 :
        Savefile = NN1_Save_Dir + '/NN1 %d/MODEL.ckpt' %iter
        saver1.save(sess, Savefile, global_step = iter)
        SavefileD = D1_Save_Dir + '/D1 %d/MODEL.ckpt' %iter
        saver_D1.save(sess, SavefileD, global_step = iter)
