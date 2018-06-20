# definition of different custom loss functions
import tensorflow as tf
import numpy as np
# Loss as defined in Fast-feature-fool


def activations(network, layers):
    loss = 0
    for i in network.keys():
        if i not in layers:
            try:
                for j in network[i].keys():
                    if j == 'concat':
                        # total blob activations
                        loss += tf.log(tf.reduce_mean(tf.abs(network[i][j])))
            except:
                # total blob activations
                loss += tf.log(tf.reduce_mean(tf.abs(network[i])))
    return loss

# Loss as defined for DG_UAP


def l2_all(network, layers):
    loss = 0
    for i in network.keys():
        if i not in layers:
            try:
                for j in network[i].keys():
                    print(i, j)
                    loss += tf.log(tf.nn.l2_loss(tf.abs(network[i][j])))
            except:
                print i
                loss += tf.log(tf.nn.l2_loss(tf.abs(network[i])))
    return loss

def l2_all_new(network, layers):
    loss = 0
    count = 0
    for i in network.keys():
        if i not in layers:
            try:
                for j in network[i].keys():
                    print(i, j)
                    loss += tf.log(tf.nn.l2_loss(tf.abs(network[i][j]))/tf.cast(tf.size(network[i][j]),tf.float32))
                    count +=1
            except:
                print i
                loss += tf.log(tf.nn.l2_loss(tf.abs(network[i]))/tf.cast(tf.size(network[i]),tf.float32))
                count += 1
    print('totalLayers',count)
    return loss/float(count)

def l2_all_new_adjusted_resnet(network, layers):
    loss = 0
    count = 0
    layers = ['conv1','pool1',
             'res2a','res2b','res2c',
             'res3a','res3b1','res3b2','res3b3','res3b4','res3b5','res3b6','res3b7',
             'res4a','res4b1','res4b2','res4b3','res4b4','res4b5','res4b6','res4b7',
             'res4b8','res4b9','res4b10','res4b11','res4b12','res4b13','res4b14','res4b15',
             'res4b16','res4b17','res4b18','res4b19','res4b20','res4b21','res4b22','res4b23',
             'res4b24','res4b25','res4b26','res4b27','res4b28','res4b29','res4b30','res4b31',
             'res4b32','res4b33','res4b34','res4b35',
              'res5a','res5b','res5c','pool5',
              'pool5_r','fc1000']
    wts = tf.placeholder(shape=[55], dtype='float32')
    for i in layers:
        print i,count
        loss += tf.log(tf.nn.l2_loss(tf.abs(network[i]))/tf.cast(tf.size(network[i]),tf.float32))*wts[count]
        count += 1
    print('totalLayers',count)
    # asdf
    return loss/float(count),wts
