#!/usr/bin/python3


######################## TRANSFER LEARNING TUTORIAL ########################

# ARGUMENTS WHEN RUNNING SCRIPT:
# ARG1: TRAIN    0: NO TRAINING    1: TRAINING
# ARG2: PREDICT  0: NO PREDICTION  1: PREDICTIONS

import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.models import save_model
#from keras.models import load_model
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array


TRAIN   = False
PREDICT = True
if (len(sys.argv)>=3):
    if (sys.argv[1]=='1'): TRAIN   = True
    if (sys.argv[2]=='0'): PREDICT = False


image_size = 224
def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    output = preprocess_input(img_array)
    return(output)


### TRANSFER LEARNING TRAINING

if (TRAIN):
    ### STEP 1: SPECIFY MODEL
    num_classes = 2
    resnet_weights_path = './input/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    
    new_model = Sequential()
    new_model.add(ResNet50(include_top=False,
                           pooling='avg',
                           weights=resnet_weights_path))
    new_model.add(Dense(num_classes, activation='softmax'))
    
    # Say not to train first layer (ResNet) model. It is already trained
    new_model.layers[0].trainable = False
    
    ### STEP 2: COMPILE MODEL
    new_model.compile(optimizer='sgd',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    
    ### STEP 3: FIT MODEL
    image_size = 224
    data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    
    train_generator = data_generator.flow_from_directory(
            './input/rural_and_urban_photos/train',
            target_size=(image_size, image_size),
            batch_size=12,
            class_mode='categorical')
    
    validation_generator = data_generator.flow_from_directory(
            './input/rural_and_urban_photos/val',
            target_size=(image_size, image_size),
            class_mode='categorical')
    
    new_model.fit_generator(
            train_generator,
            steps_per_epoch=6,
            validation_data=validation_generator,
            validation_steps=1)
    
    #new_model.summary()
    save_model(new_model, "./input/tl_model_urban_vs_rural.h5")


### PREDICTIONS

if (PREDICT):
    new_model = load_model('./input/tl_model_urban_vs_rural.h5')
    test_dir = "./input/rural_and_urban_photos/test/"
    test_size = 5 
    city_paths  = [test_dir + "v" + str(i) + ".jpg" for i  in range(1,test_size+1)]
    rural_paths = [test_dir + "r" + str(i) + ".jpg" for i  in range(1,test_size+1)]
    
    img_paths = city_paths + rural_paths
    
    test_data = read_and_prep_images(img_paths)
    preds = new_model.predict(test_data)
    print(preds)
    
    plt.figure()
    i=0
    for img_path in img_paths:
        i+=1
        plt.subplot(2,test_size,i)
        img=mpimg.imread(img_path)
        plt.imshow(img) 
    
    plt.show()

