from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#imporing Libraries
import tensorflow as tf
import pandas as pd 
import numpy as np 
import os
from skimage import data, io, filters
from skimage.transform import *
import random

from  argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--lr",action="store", dest="lr",default = 0.0005,type= float)
parser.add_argument("--batch_size",action="store", dest="batch_size",default = 5,type=int)
parser.add_argument("--init",action="store", dest="intiMethod",default = 2,type=int)
parser.add_argument("--save_dir",action="store",dest="save_dir",default="sav/")
parser.add_argument("--expt_dir",action="store",dest="expt_dir",default="")
parser.add_argument("--train",action="store",dest="train",default="train.csv")
parser.add_argument("--val",action="store",dest="val",default="val.csv")
parser.add_argument("--test",action="store",dest="test",default="test.csv")
parser.add_argument("--epochs",action="store",dest="epochs",default=7,type=int)
parser.add_argument("--l1reg",action="store", dest="l1reg",default = 0.0,type= float)
parser.add_argument("--l2reg",action="store", dest="l2reg",default = 0.0,type= float)
parser.add_argument("--valcpt",action="store", dest="valcpt",default = 100,type= int)
parser.add_argument("--patientL",action="store", dest="pl",default = 5,type= int)
parser.add_argument("--dataAug",action="store", dest="dataAug",default = 0,type= int)


args = parser.parse_args()



phaseTrain = True

#Setting Seed
tf.set_random_seed(1234)

#Weight Regularization
l1regScale = args.l1reg
l2regScale = args.l2reg

#Setting Hyper Paramter
batch_size = args.batch_size
lr = args.lr
initMethod = args.intiMethod
valSetSize = 5
epochs = args.epochs
nonlnrty = tf.nn.relu
validationChkPoint = args.valcpt #After how many steps you wish to see validation performance
#saveDir = "/home/sp/deepLearning/pa3/"
saveDir = args.save_dir
patienceLimit = args.pl #For early stopping.

#To generate AugmentedData
genAugmentedData = args.dataAug

#Opening And Writing Heading in the log file
fileName = saveDir+"e_" + str(epochs) + "_b_" + str(batch_size) + "_lr_" + str(lr) + "_initMethod_" + str(initMethod) + "save.csv"
f = open(fileName,"w")
f.write("e, s, tr_e, tr_a, v_e, v_a\n")

#Dataset Path
#datasetLoc = "/home/sp/deepLearning/pa3/"
datasetLoc = args.expt_dir
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

ValDnp = None

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

def trainDataNormalizer(unNormdata):
    tiled_mn = np.tile(mn, (no_of_row, 1))
    tiled_st_dev = np.tile(st_dev, (no_of_row, 1))
    mn_shifted_data = unNormdata - tiled_mn
    unNormdata = mn_shifted_data/tiled_st_dev
    return unNormdata #Now Normalized

#Setting All the useless variables to None so that there memory can be freed

#Cleaning Area###############################

tiled_mn = None
val_tiled_mn = None
test_tiled_mn = None

tiled_st_dev = None 
val_tiled_st_dev = None
test_tiled_st_dev = None

mn_shifted_data = None 
mn_shifted_val_d = None
mn_shifted_test_d = None

trainD = None
valD = None
testD = None

########################################################################################
################# Support Code ########################################################
######################################################################################

def transformOp(image):
    img = np.reshape(image,(28,28))
    x1 = random.randint(-3,3)
    y1 = random.randint(-3,3)
    angle = random.randint(-40,40)
    flp = random.randint(0,1)
    form = SimilarityTransform(translation=(x1, y1))
    img = warp(img, form,preserve_range=True)
    img = rotate(img,angle,preserve_range=True)
    if flp == 1:
        img = img[:, ::-1]
    return np.reshape(img,(784,))

##############################################

#Data set Variables are trainFeatures, valFeatures, testFeatures and truth valLabels, trainLabels

###################################################
#Start of the Graph TF ############################
########################################3#############

# Declaring Placeholders
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
    activation=nonlnrty,
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
    activation = nonlnrty,
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
    activation = nonlnrty,
    kernel_initializer = initializerUsed,name = 'conv3',
    kernel_regularizer= reglr,
    bias_regularizer = reglr
      )

#convo layer 4
conv4 = tf.layers.conv2d( inputs = conv3,
    filters = 256,
    kernel_size = [3,3],  
    padding = "same",
    activation = nonlnrty,
    kernel_initializer = initializerUsed ,name = 'conv4',
    kernel_regularizer= reglr,
    bias_regularizer = reglr
     )

#pool layer 3
pool3 = tf.layers.max_pooling2d(inputs = conv4,padding="same",pool_size = [2,2],strides = 1,name = 'pool3')

#Making pool3 flat for the purpose of sending it to dense layer
pool3_flat = tf.reshape(pool3,[-1,28*28*256])

fc1 = tf.layers.dense(inputs=pool3_flat, units=1024, activation=nonlnrty,kernel_initializer = initializerUsed,kernel_regularizer = reglr,
    bias_regularizer = reglr,name = 'fc1')

#fully connected 2
fc2 = tf.layers.dense(inputs=fc1,units = 1024,activation =nonlnrty,kernel_regularizer= reglr,
    bias_regularizer = reglr,kernel_initializer = initializerUsed,name = 'fc2')


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
    training=phaseTrain,
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

#Last Layer // Output Layer
logits = tf.layers.dense(inputs = normalizedBatch,units =10,kernel_initializer = initializerUsed,name = 'logits',kernel_regularizer= reglr,
    bias_regularizer = reglr)

#Getting Prediction Labels as tensor 1d
predictedLabels = tf.argmax(input=logits, axis=1)
pll = tf.cast(predictedLabels,tf.int32)
c = tf.equal(pll,labels[:,0]) 
correctCount = tf.reduce_sum(tf.cast(c, tf.int32)) #Getting the count of correct predicition

#For Training Core : one_hot, Loss, optimizer, training
onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
loss = tf.losses.softmax_cross_entropy(onehot_labels,logits)
tf.control_dependencies(update_ops)
optimizer = tf.train.RMSPropOptimizer(lr,momentum=0.08)
training = optimizer.minimize(loss)

#For saving the graph Variables
saver = tf.train.Saver()

#Declaring the session
sess = tf.Session()
sess.run(tf.global_variables_initializer()) #Initializing the session


#Not applying any restriction since data size is 55000, and batch size will be multiple of 5
# These condition are by choice.
totalSteps = ((np.shape(trainFeatures))[0])//batch_size
validataionSize = len(valFeatures)

#For early stopping 
to_check_with = 1.0 #Initializing Min error for eary stopping
current_patience = 0 #Initializing current patience with 0

# cc,t,p = sess.run([correctCount,loss,predictedLabels],feed_dict={features:trainFeatures[:5],labels:trainLabels[:5]})
# print(t," ",p," ",trainLabels[:5,0])
# print(cc)
AugTrainFeature = None
for i in range(0,epochs): # The no. time whole data should be passed for training 
    phaseTrain = True
    if genAugmentedData == 1:
        if i != 1:
            AugTrainFeature = np.zeros((55000,784))
            for icount in range(len(trainFeatures)):
                image = trainFeatures[icount]
                AugTrainFeature[icount] = transformOp(image)
            AugTrainFeature = trainDataNormalizer(AugTrainFeature)
        else:
            AugTrainFeature = trainFeatures
    else:
        AugTrainFeature = trainFeatures
    correctP = 0
    total = 0
    totalLoss = 0
    for step in range(0,totalSteps):
        phaseTrain = False
        #Retriving a Batch for process
        batchD = AugTrainFeature[(step*batch_size):((step+1)*batch_size)] 
        batchL = trainLabels[(step*batch_size):((step+1)*batch_size)]

        print(step)

        #Running the Heart
        _,_,t,TcorrectP = sess.run([update_ops,training,loss,correctCount],feed_dict={features:batchD,labels:batchL})
        
        batchD = None
        
        #Updating the counts and total variables
        correctP = correctP + TcorrectP
        total = total + batch_size
        totalLoss = totalLoss + t
        
        batchL = None
        
        #For the calculation of validation accuracy
        if (step%validationChkPoint == 0 and step!=0) or step==totalSteps-1:
            CpredVal = 0
            valTLoss = 0
            total_ValSteps = int(validataionSize//valSetSize) 
            #getting Loss and Accuracy on the validation Data
            
            for valStep in range(total_ValSteps):
                print("Val Step : ",valStep,"\n")
                vf = valFeatures[(valStep*valSetSize):((valStep+1)*valSetSize)] 
                vll = valLabels[(valStep*valSetSize):((valStep+1)*valSetSize)]
                valLoss,VcorrectP = sess.run([loss,correctCount],feed_dict={features:vf,labels:vll})
                valTLoss = valTLoss + valLoss
                CpredVal = CpredVal + VcorrectP
            vll = None
            vf = None
            
            #Calculating Validation Accuracy
            accuracyVal = CpredVal/validataionSize

            #Calculating Accuracy on the Training Data
            accuracyTrain = correctP/total
            
            #Normalizing Training Loss
            totalLoss = totalLoss/validationChkPoint
            
            #Normalizing Validation Loss
            valTLoss = valTLoss/total_ValSteps

            #Writing to CSV log file
            print("e ",i," s ",step," tr l ",totalLoss," tr acc ",accuracyTrain," Val l ",valTLoss," Val A ",accuracyVal)
            f.write(str(i)+", "+str(step)+", "+str(totalLoss)+", "+str(accuracyTrain)+", "+str(valTLoss)+", "+str(accuracyVal)+"\n")
            f.flush()

            #For saving the models
            #if accuracyVal > 0.92:
            #    cpName = saveDir + "model_e_" + str(i)+"_s_"+str(step) +".ckpt"
            #   saver.save(sess,cpName)
            # Removed due to early stoping


            val_error = 1 - accuracyVal
    
            if val_error > to_check_with:               #it is not favourable
                current_patience = current_patience + 1
                if(current_patience >= patienceLimit):              # we have reached our patience 
                    print("Patience level exceeded for below threshold vlues")
                    exit()
            else:
                to_check_with = val_error   #if current error is less than previous
                current_patience = 0
                if val_error < 0.12:
                    os.system("rm "+ saveDir + "model_e_*")
                    chckmodel = saveDir + "model_e_" + str(i)+"_s_"+str(step) +".ckpt"
                    saver.save(sess,chckmodel)


            #Resetting Paramter : Making way for new information
            total = 0
            correctP = 0
            totalLoss = 0

f.close()
