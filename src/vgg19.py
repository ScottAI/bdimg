#coding=utf8
import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt

from keras import __version__
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Dropout,Activation,Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import getdata
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.6
#set_session(tf.Session(config=config))
FC_SIZE = 1024
NB_IV3_LAYERS_TO_FREEZE = 25
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def get_nb_files(directory):
  """Get number of files by searching directory recursively"""
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
  return cnt


def setup_to_transfer_learn(model, base_model):
  """Freeze all layers and compile the model"""
  for layer in base_model.layers:
    layer.trainable = False
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


def add_new_last_layer(base_model, nb_classes):
  """
  迁移学习
  """
  x = base_model.output
  x = Flatten(input_shape=base_model.output_shape[1:])(x)
  x = Dense(FC_SIZE)(x) #new FC layer, random init
  x = Activation('relu')(x)
  x = Dropout(0.5)(x)
  predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
  model = Model(inputs=base_model.input, outputs=predictions)
  return model


def setup_to_finetune(model):
  """finetune
  """
  for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
     layer.trainable = False
  for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
     layer.trainable = True
  model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


def train(X_train,Y_train,X_test,Y_test,depoch=50,ftepoch=201,batch_size=32,classnum=100,out='inceptionv3-ft.model'):
  """Use transfer learning and fine-tuning to train a network on a new dataset"""
  nb_train_samples = len(Y_train)
  nb_classes = classnum
  nb_val_samples = len(Y_test)
  batch_size = batch_size

  # data prep
  train_datagen =  ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range=20,#角度
      width_shift_range=0.2,#水平偏移
      height_shift_range=0.2,#高度偏移
      shear_range=0.1,#剪切强度,逆时针方向的剪切变化角度
      zoom_range=0.2,#随机缩放的幅度
      horizontal_flip=True,#进行随机水平反转
      vertical_flip=False#进行竖直反转
  )
  test_datagen = ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.1,
      zoom_range=0.2,
      horizontal_flip=True
  )

  train_generator = train_datagen.flow(X_train, Y_train, batch_size=batch_size, seed=42)

  validation_generator = test_datagen.flow(X_test, Y_test, batch_size=batch_size, seed=42)
  X_test = preprocess_input(X_test)
  # setup model
  base_model = VGG19(weights='imagenet', include_top=False,input_shape=(224,224,3)) #include_top=False excludes final FC layer
  model = add_new_last_layer(base_model, nb_classes)

  # transfer learning
  setup_to_transfer_learn(model, base_model)
  for i in range(depoch):
    print('Epoch: ',i)
    model.fit_generator(train_generator,epochs=1,
                    steps_per_epoch = int(nb_train_samples/batch_size))
    #score, acc = model.evaluate_generator(validation_generator,int(nb_val_samples/batch_size),workers=30,max_q_size=100)
    #print('epoch: ',i,' val_acc: ',acc)
    score1, acc1 = model.evaluate(X_test, Y_test, batch_size=batch_size)
    print('epoch: ',i,'eval_acc: ',acc1)

  # fine-tuning
  setup_to_finetune(model)
  for i in range(ftepoch):
    print('Epoch: ',i)
    model.fit_generator(train_generator,epochs=1,
                    steps_per_epoch = int(nb_train_samples/batch_size))
    #score,acc = model.evaluate_generator(validation_generator,int(nb_val_samples/batch_size),workers=30,max_q_size=100)
    #print('epoch: ',i,' val_acc: ',acc)
    score1, acc1 = model.evaluate(X_test, Y_test, batch_size=batch_size)
    print('epoch: ',i,'eval_acc: ',acc1)
    if i%10 == 0 and i !=0:
      model.save(out+str(i))
  #X_test = preprocess_input(X_test)
  score, acc = model.evaluate(X_test, Y_test, batch_size=batch_size)
  print('now accu:',acc)
  print('ALL DONE')


def makeonehot(X,dim):
  res = []
  for i in X:
    here = np.zeros(dim)
    here[i]=1
    res.append(here)
  res = np.asarray(res)
  return res

if __name__=="__main__":
  classnum = 99
  X,Y = getdata.get('/home/lihang/2017/bdimg/data/train_data2/',224,224,3)
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.93, random_state=42)
  Y_train = makeonehot(Y_train,classnum)
  Y_test = makeonehot(Y_test,classnum)
  train(X_train,Y_train,X_test,Y_test,depoch=25,ftepoch=201,batch_size=32,classnum=classnum,out='2inception.model')
