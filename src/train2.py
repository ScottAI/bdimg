#coding=utf8
import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt

from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Dropout,Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import getdata
import numpy as np

FC_SIZE = 1024
NB_IV3_LAYERS_TO_FREEZE = 172
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
  x = GlobalAveragePooling2D()(x)
  x = Dense(FC_SIZE)(x) #new FC layer, random init
  x = Dropout(0.5)(x)
  x = Activation('relu')(x)
  predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
  model = Model(inputs=base_model.input, outputs=predictions)
  return model


def setup_to_finetune(model):
  """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.

  note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch

  Args:
    model: keras model
  """
  for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
     layer.trainable = False
  for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
     layer.trainable = True
  model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


def train(X_train,Y_train,X_test,Y_test,epoch=100,batch_size=32,out='inceptionv3-ft.model'):
  """Use transfer learning and fine-tuning to train a network on a new dataset"""
  nb_train_samples = len(Y_train)
  nb_classes = 100
  #nb_val_samples = len(Y_test)
  nb_epoch = epoch
  batch_size = batch_size

  # data prep
  train_datagen =  ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range=10,#角度
      width_shift_range=0.1,#水平偏移
      height_shift_range=0.1,#高度偏移
      shear_range=0.1,#剪切强度,逆时针方向的剪切变化角度
      zoom_range=0.1,#随机缩放的幅度
      horizontal_flip=True,#进行随机水平反转
      vertical_flip=False#进行竖直反转
  )
  test_datagen = ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range=10,
      width_shift_range=0.1,
      height_shift_range=0.1,
      shear_range=0.1,
      zoom_range=0.1,
      horizontal_flip=True
  )

  train_generator = train_datagen.flow(X_train, Y_train, batch_size=batch_size, seed=42)

  #validation_generator = test_datagen.flow(X_test, Y_test, batch_size=batch_size, seed=42)

  # setup model
  base_model = InceptionV3(weights='imagenet', include_top=False) #include_top=False excludes final FC layer
  model = add_new_last_layer(base_model, nb_classes)

  # transfer learning
  setup_to_transfer_learn(model, base_model)

  history_tl = model.fit_generator(
    train_generator,
    epochs=30,
    steps_per_epoch = int(nb_train_samples/batch_size),
    #validation_data=(X_test,Y_test),
    #validation_data=validation_generator,
    #validation_steps=nb_val_samples,
    class_weight='auto',
    workers=10,
    max_q_size=50
    )

  # fine-tuning
  setup_to_finetune(model)

  history_ft = model.fit_generator(
    train_generator,
    epochs=100,
    steps_per_epoch = int(nb_train_samples/batch_size),
    #validation_data=(X_test,Y_test),
    #validation_data=validation_generator,
    #validation_steps=nb_val_samples,
    class_weight='auto',
    workers=10,
    max_q_size=50
    )

  model.save(out)
 # X_test = preprocess_input(X_test)
  #score, acc = model.evaluate(X_test, Y_test, batch_size=batch_size)
  #print('now accu:',acc)
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
  X,Y = getdata.get('/home/lihang/2017/bdimg/data/train_data/')
  #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.9, random_state=33)
  Y_train = makeonehot(Y_train,100)
  #Y_test = makeonehot(Y_test,100)
  train(X,Y_train,None,None,epoch=20,batch_size=32,out='inceptionv3-ft.model')
