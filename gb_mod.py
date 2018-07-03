
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import tensorflow as tf
import pandas as pd 
import numpy as np 
import os
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

indices =[[0,0,7,22,189]]
mx = 16
tf.set_random_seed(1234)
#Pickle for saving Data

#Pickle Command for the purpose of saving object in file and reloading
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name, 'rb') as f:
        return pickle.load(f)



l1regScale = 0.5
l2regScale = 0.5

i = 9
step = 100

batch_size = 10
lr = 0.0001
initMethod = 2
valSetSize = 10
epochs = 1
saveDir = "/home/ubuntu/pa3/saveDr/"


chkLocation = saveDir+"model_e_" + str(i)+"_s_"+str(step) +".ckpt"

fileName = saveDir+"submission.csv"

f = open(fileName,"w")

f.write("pixelno,grad\n")

datasetLoc = "/home/ubuntu/pa3/"

#Reading Data
trainD = pd.read_csv( datasetLoc + "train.csv", low_memory=False)
valD = pd.read_csv( datasetLoc + "val.csv", low_memory=False)
testD = pd.read_csv( datasetLoc + "test.csv", low_memory=False)

#Breaking Labels and Features

#Training
trainDnp = trainD.values
trainLabels = trainDnp[0:55001,785:786] #Label Classification
trainFeatures = trainDnp[0:55000,1:785] #Traning Input data
trainDnp = None

#Validation
valDnp = valD.values
valLabels = valDnp[0:5001,785:786] #Label Classification
valFeatures = valDnp[0:5000,1:785] #Validation Input Data

valDnp = None

#Testing
testDnp = testD.values
testFeatures = testDnp[:,1:785] #Testing we don't have Labels

testDnp = None
## Normalizing Data

(no_of_row,no_of_col) = (np.shape(trainFeatures)) #Training Operation
(val_row,val_col) = (np.shape(valFeatures)) #Validation Operation
(test_row,test_col) = (np.shape(testFeatures)) #Validation Operation

mn = np.mean(trainFeatures,axis = 0) #Mean of training data
tiled_mn = np.tile(mn, (no_of_row, 1)) #Training : tiling up mean for substraction
val_tiled_mn = np.tile(mn,(val_row,1)) #Validation : tiling up mean for validation
test_tiled_mn = np.tile(mn,(test_row,1)) #Testing : tiling up mean for validation

st_dev = np.std(trainFeatures,axis = 0) #Standard deviation of training data
tiled_st_dev = np.tile(st_dev, (no_of_row, 1)) #Training : tiling of variance for division
val_tiled_st_dev = np.tile(st_dev,(val_row,1)) #Validation : tiling up variance for division
test_tiled_st_dev = np.tile(st_dev,(test_row,1)) #Testing : tiling up variance for division

#Normalizing Training Data
mn_shifted_data = trainFeatures - tiled_mn 
trainFeatures = mn_shifted_data/tiled_st_dev 

#Normalizing Validation Data
mn_shifted_val_d = valFeatures - val_tiled_mn
valFeatures = mn_shifted_val_d/val_tiled_st_dev

#Normalizing Testing Data
mn_shifted_test_d = testFeatures - test_tiled_mn
testFeatures = mn_shifted_test_d/test_tiled_st_dev

#Setting All the useless variables to None so that there memory can be freed

#Cleaning Area###############################

mn = None
tiled_mn = None
val_tiled_mn = None
test_tiled_mn = None

st_dev = None
tiled_st_dev = None 
val_tiled_st_dev = None
test_tiled_st_dev = None

mn_shifted_data = None 
mn_shifted_val_d = None
mn_shifted_test_d = None

trainD = None
valD = None
testD = None

##############################################

#Data set Variables are trainFeatures, valFeatures, testFeatures and truth valLabels, trainLabels

features = tf.placeholder(dtype=tf.float32)
labels = tf.placeholder(dtype=tf.int32)

#Initialization Method Deciding Condition
if initMethod == 1:
    initializerUsed = tf.contrib.layers.xavier_initializer()
else:
    initializerUsed = tf.keras.initializers.he_normal()


# for input layer 
input_layer =tf.reshape(features,[-1,28,28,1]) # here 3 corresponds to RGB channel

reglr = tf.contrib.layers.l1_l2_regularizer(
    scale_l1=l1regScale,
    scale_l2=l2regScale,
    scope=None
)

#conv layer 1 
conv1 = tf.layers.conv2d(
    inputs = input_layer,
    filters = 64,
    kernel_size = [3,3],
    padding = "same",
    activation=tf.nn.relu,
    kernel_initializer = initializerUsed,
    name = 'conv1',
    kernel_regularizer= reglr,
    bias_regularizer = reglr
    )

# #Pooling Layer 1
pool1 = tf.layers.max_pooling2d(inputs = conv1,padding="same",pool_size = [2,2],strides = 1,name = 'pool1')

#conv layer 2
conv2 = tf.layers.conv2d(
    inputs = pool1,
    filters = 128,
    kernel_size = [3,3],  
    padding = "same",
    activation = tf.nn.relu,
    kernel_initializer = initializerUsed,
    name='conv2',
    kernel_regularizer= reglr,
    bias_regularizer = reglr
      )

#pooling layer 2
pool2 = tf.layers.max_pooling2d(inputs = conv2,padding="same",pool_size = [2,2],strides = 1,name ='pool2')

#convo layer 3
conv3 = tf.layers.conv2d(
    inputs = pool2,
    filters = 256,
    kernel_size = [3,3],  
    padding = "same",
    activation = tf.nn.relu,
    kernel_initializer = initializerUsed,name = 'conv3',
    kernel_regularizer= reglr,
    bias_regularizer = reglr  )

#convo layer 4
conv4 = tf.layers.conv2d( inputs = conv3,
    filters = 256,
    kernel_size = [3,3],  
    padding = "same",
    activation = tf.nn.relu,
    kernel_initializer = initializerUsed, name = 'conv4',
    kernel_regularizer= reglr,
    bias_regularizer = reglr  )

#pool layer 3
pool3 = tf.layers.max_pooling2d(inputs = conv4,padding="same",pool_size = [2,2],strides = 1,name = 'pool3')

#Making pool3 flat for the purpose of sending it to dense layer
pool3_flat = tf.reshape(pool3,[-1,28*28*256])

fc1 = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu,kernel_initializer = initializerUsed,name = 'fc1',kernel_regularizer= reglr,
    bias_regularizer = reglr)

#fully connected 2
fc2 = tf.layers.dense(inputs=fc1,units = 1024,activation =tf.nn.relu,kernel_initializer = initializerUsed,name = 'fc2',kernel_regularizer= reglr,
    bias_regularizer = reglr)


# ##############BATCH NORMALISATION IS LEFT HERE ####################################
normalizedBatch = tf.layers.batch_normalization(
    inputs=fc2,
    axis=-1,
    momentum=0.99,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer=tf.zeros_initializer(),
    gamma_initializer=tf.ones_initializer(),
    moving_mean_initializer=tf.zeros_initializer(),
    moving_variance_initializer=tf.ones_initializer(),
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    training=False,
    trainable=True,
    name=None,
    reuse=None,
    renorm=False,
    renorm_clipping=None,
    renorm_momentum=0.99,
    fused=None,
    virtual_batch_size=None,
    adjustment=None
)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)


logits = tf.layers.dense(inputs =normalizedBatch,units =10,kernel_initializer = initializerUsed,name = 'logits')
predictedLabels = tf.argmax(input=logits, axis=1)
pll = tf.cast(predictedLabels,tf.int32)
c = tf.equal(pll,labels[:,0])

correctCount = tf.reduce_sum(tf.cast(c, tf.int32))
onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
loss = tf.losses.softmax_cross_entropy(onehot_labels,logits)
optimizer = tf.train.AdamOptimizer(lr)
training = optimizer.minimize(loss)

#out4 = np.asarray(conv4)
#only1 = np.zeros([1,1,28,28,256])
#only1[0,0,:,0,0] =1.0
#masked = np.multiply(out4,only1)
#only2 = tf.convert_to_tensor(only1, np.float32)
#masked = tf.multiply(conv4, tf.cast(only2,dtype = tf.float32))
c = tf.constant(0.0,shape = [1,1,28,28,256])
#indices =[[0,0,9,14,219]]
values = [1.0]
shape = [1,1,28,28,256]
delta = tf.SparseTensor(indices, values, shape)	
result = c + tf.sparse_tensor_to_dense(delta)
masked = tf.multiply(conv4, result)




#grads_wrt_input_tensor = tf.gradients(masked, features)[0]
grad = tf.gradients(masked, features)


saver = tf.train.Saver()
sess = tf.Session()
#sess : Session, chkLocation : Location of checkpoint
saver.restore(sess,chkLocation)

batchD = trainFeatures[1]   # this is t-shirt

grade,cv4= sess.run([grad,conv4],feed_dict={features:batchD})
#maxm = np.amax(cv4)
#index = [np.concatenate(np.where(cv4 ==maxm)) ]
#print(type(cv4))
#print(np.shape(cv4))
#print(cv4)
print(mn)

save_obj(grade,"grad"+str(mx))




#grade = sess.run([grad],fee_dict ={features:batchD,indices:index})
#print(cv4[0,27,27,1])
#print(grade)
#cv4= sess.run([conv4],feed_dict={features:batchD})
# find  index of max 25 elements 
#ind = np.argpartition(cv4, -25)[-25:]
#max,indices] = tf.nn.top_k(cv4,k=25,sorted=True)
#print(ind[0,1])
#for i in range(25):
   # grade = sess.run([grad],feed_dict  ={indices = max[i]})
   # save_obj(grade,"gradients_mod"+i)





#save_obj(cv4,"cv4")



