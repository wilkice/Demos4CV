"""input a captcha img, return the valid code for it
author: Jiabin Yan
"""
import os

import tensorflow as tf
import cv2
import pandas as pd
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.app.flags.DEFINE_integer('batch_size', 10,'how many items in each batch')
# tf.app.flags.DEFINE_integer('capacity', 20, 'how many batchs')
FLAGS = tf.app.flags.FLAGS

def filename2num(filename,df):
    labels = []
    for file in filename:
        labels.append(df[df['img_name']==file]['char2num'])
    return np.array(labels)

def img_read(img_path):
    img_list = [os.path.join(img_path, file) for file in os.listdir(img_path)]
    filename_queue = tf.train.string_input_producer(img_list)

    # value: metadata for img
    filename, value = tf.WholeFileReader().read(filename_queue)
    img = tf.image.decode_jpeg(value)

    img.set_shape([60, 160, 3])

    
    # because our img is the same size, so we don't need to resize


    img_batch, filename_batch = tf.train.batch([img, filename], batch_size=16,
                                                num_threads=2,
                                                capacity=32)
    
                                                # batch_size = FLAGS.batch_size, 
                                                # num_threads=2) 
                                                # capacity= FLAGS.capacity)
    return img_batch, filename_batch

def weight_var(shape, name=None):
    return tf.Variable(tf.truncated_normal(shape, mean=0.0, stddev=0.01,
                                            dtype=tf.float32),name=name)
def bias_var(shape, name=None):
    return tf.Variable(tf.zeros(shape, dtype=tf.float32), name=name)

def model():
    """create cnn model
    """
    with tf.variable_scope('data'):
        x = tf.placeholder(tf.float32, [None, 60 , 160, 3])
        y_true = tf.placeholder(tf.float32, [None, 4*62])


    with tf.variable_scope('conv1'):
        w_conv1 = weight_var([3, 3, 3, 32], name='w_conv1')
        b_conv1 = bias_var([32], name='b_conv1')
        x_conv1 = tf.nn.conv2d(x, filter=w_conv1, strides=[1,1,1,1], 
                                padding='SAME', name='conv1_2d') + b_conv1

        x_relu1 = tf.nn.relu(x_conv1, name='relu1')
        x_pool1= tf.nn.max_pool(x_relu1, ksize=[1,2,2,1], strides=[1,2,2,1],
                                padding='SAME', name='pool1')

    with tf.variable_scope('conv2'):
        w_conv2 = weight_var([3, 3, 32, 64], name='w_conv2')
        b_conv2 = bias_var([64], name='b_conv2')
        x_conv2 = tf.nn.conv2d(x_pool1, filter=w_conv2, strides=[1,1,1,1], 
                                padding='SAME', name='conv2_2d') + b_conv2
                                
        x_relu2 = tf.nn.relu(x_conv2, name='relu2')
        x_pool2= tf.nn.max_pool(x_relu2, ksize=[1,2,2,1], strides=[1,2,2,1],
                                padding='SAME', name='pool1')
        

    with tf.variable_scope('fc'):
        w_fc = weight_var([15*40*64, 4*62], name='w_fc')
        b_fc = bias_var([4*62])

        logits = tf.matmul(tf.reshape(x_pool2, [-1, 15*40*64]), w_fc) + b_fc
    
    return x, y_true, logits



def captcha(csv_file='char2num.csv', img_path='./cap_img'):
    """
    read img using queue
    """
    df = pd.read_csv(csv_file)
    
    img_batch, filename_batch = img_read(img_path)
    x, y_true, logits = model()
    
    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=logits))
    with tf.variable_scope('optimize'):
        train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
    with tf.variable_scope('accuracy'):
        equal_list = tf.reduce_all(
            tf.equal(tf.argmax(tf.reshape(logits, [-1, 4, 62]), axis=-1),
                    tf.argmax(tf.reshape(y_true, [-1,4,62]),axis=-1)),
                    axis=-1
        )
        accuracy=tf.reduce_mean(tf.cast(equal_list, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(sess, coord)
        if os.path.exists('./models/captcha/checkpoint'):
            saver.restore(sess, './models/captcha/captcha')
        for i in range(200):
            imgs, filenames = sess.run([img_batch, filename_batch])
            labels = filename2num(filenames, df)
            labels_onehot = tf.reshape(tf.one_hot(labels, 62), [-1, 4*62]).eval()
            _, loss_val, acc = sess.run([train_op, loss, accuracy], 
                                    feed_dict={x:imgs, y_true:labels_onehot})
            print('times: {}, loss: {}, acc: {}'.format(i,loss_val,acc))
            if (i+1) % 50 ==0:
                saver.save(sess, './models/captcha/captcha')
        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    captcha()