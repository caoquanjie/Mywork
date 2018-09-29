import tensorflow as tf
import numpy as np
import utils
"""
load variable from npy to build the VGG
:param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
"""
data_dict = np.load('vgg19.npy', encoding='latin1').item()
def Vgg19(rgb):
    rgb_scaled = rgb*255.0
    VGG_MEAN = [103.939, 116.779, 123.68]
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
    assert red.get_shape().as_list()[1:] == [224, 224, 1]
    assert green.get_shape().as_list()[1:] == [224, 224, 1]
    assert blue.get_shape().as_list()[1:] == [224, 224, 1]
    bgr = tf.concat(axis=3, values=[
        blue - VGG_MEAN[0],
        green - VGG_MEAN[1],
        red - VGG_MEAN[2],
    ])
    assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
    conv1_1 = conv_layer(bgr, "conv1_1", w_shape=[3,3,3,64], b_shape=[64,])
    conv1_2 = conv_layer(conv1_1, "conv1_2", w_shape=[3,3,64,64], b_shape=[64,])
    pool1 = max_pool(conv1_2, 'pool1')

    conv2_1 = conv_layer(pool1, "conv2_1", w_shape=[3,3,64,128], b_shape=[128,])
    conv2_2 = conv_layer(conv2_1, "conv2_2",w_shape=[3,3,128,128], b_shape=[128,])
    pool2 = max_pool(conv2_2, 'pool2')

    conv3_1 = conv_layer(pool2, "conv3_1",w_shape=[3,3,128,256], b_shape=[256])
    conv3_2 = conv_layer(conv3_1, "conv3_2",w_shape=[3,3,256,256], b_shape=[256,])
    conv3_3 = conv_layer(conv3_2, "conv3_3",w_shape=[3,3,256,256], b_shape=[256,])
    conv3_4 = conv_layer(conv3_3, "conv3_4",w_shape=[3,3,256,256], b_shape=[256,])
    pool3 = max_pool(conv3_4, 'pool3')

    conv4_1 = conv_layer(pool3, "conv4_1",w_shape=[3,3,256,512], b_shape=[512,])
    conv4_2 = conv_layer(conv4_1, "conv4_2",w_shape=[3,3,512,512], b_shape=[512,])
    conv4_3 = conv_layer(conv4_2, "conv4_3",w_shape=[3,3,512,512], b_shape=[512,])
    conv4_4 = conv_layer(conv4_3, "conv4_4",w_shape=[3,3,512,512], b_shape=[512,])
    pool4 = max_pool(conv4_4, 'pool4')

    conv5_1 = conv_layer(pool4, "conv5_1",w_shape=[3,3,512,512], b_shape=[512,])
    conv5_2 = conv_layer(conv5_1, "conv5_2",w_shape=[3,3,512,512], b_shape=[512,])
    conv5_3 = conv_layer(conv5_2, "conv5_3",w_shape=[3,3,512,512], b_shape=[512,])
    conv5_4 = conv_layer(conv5_3, "conv5_4",w_shape=[3,3,512,512], b_shape=[512,])
    pool5 = max_pool(conv5_4, 'pool5')

    fc6 = fc_layer(pool5, "fc6",w_shape=[25088,4096],b_shape=[4096,])
    assert fc6.get_shape().as_list()[1:] == [4096]
    relu6 = tf.nn.relu(fc6)

    logit = fc_layer(relu6, "fc7",w_shape=[4096,4096],b_shape=[4096,])
    #logit = tf.nn.relu(fc7)
    return logit

    #fc8 = fc_layer(relu7, "fc8",w_shape=[4096,1000],b_shape=[1000])

    #prob = tf.nn.softmax(fc8, name="prob")
    #return prob


def avg_pool(bottom, name):
    return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def max_pool(bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def conv_layer(bottom, name, w_shape,b_shape):
    with tf.variable_scope(name):
        filt = get_conv_filter(name, w_shape)

        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

        conv_biases = get_bias(name,b_shape)
        bias = tf.nn.bias_add(conv, conv_biases)

        relu = tf.nn.relu(bias)
        return relu


def fc_layer(bottom, name,w_shape,b_shape):
    with tf.variable_scope(name):
        shape = bottom.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(bottom, [-1, dim])

        weights = get_fc_weight(name,w_shape)
        biases = get_bias(name,b_shape)

        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

        return fc


def get_conv_filter(name, shape):
    init=data_dict[name][0]
    return tf.get_variable(name+'_W', dtype=tf.float32,shape=shape,
                           initializer=tf.constant_initializer(init))


def get_bias(name, shape):
    init = data_dict[name][1]
    return tf.get_variable(name+'_b', dtype=tf.float32, shape=shape,
                           initializer=tf.constant_initializer(init))


def get_fc_weight(name,shape):
    init=data_dict[name][0]
    return tf.get_variable(name+'_W', dtype=tf.float32,shape=shape,
                           initializer=tf.constant_initializer(init))



images=tf.placeholder(dtype=tf.float32,shape=[1,224,224,3],name='images')
logit_arr=[]
'''with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    for i in range(5):
        logit_arr.append(Vgg19(images))'''

var_list = tf.trainable_variables()
logit_var = Vgg19(images)
with tf.Session() as sess:
    # input images
    img1 = utils.load_image("./test_data/bird_1.jpg")
    #img2 = utils.load_image("./test_data/bird_2.jpg")
    batch1 = img1.reshape((1, 224, 224, 3))

    #batch = np.concatenate((batch1, batch2), 0)
    sess.run(tf.global_variables_initializer())
    logit_val = sess.run(logit_var,feed_dict={images:batch1})
    print(logit_val)
    print(logit_val.shape)