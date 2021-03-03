#!/usr/bin/python3


################################ DEEP LEANRING - BUILD MODEL ################################

# ARGUMENTS WHEN RUNNING SCRIPT:
# ARG1: TRAIN    0: NO TRAINING    1: TRAINING
# ARG2: PREDICT  0: NO PREDICTION  1: PREDICTIONS

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout
from tensorflow.keras.models import save_model
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input


### Define constants

TRAIN   = False
PREDICT = True
if (len(sys.argv)>=3):
    if (sys.argv[1]=='1'): TRAIN   = True
    if (sys.argv[2]=='0'): PREDICT = False

img_rows, img_cols = 28, 28
num_classes = 10


### Define useful functions

def data_prep(raw):
    out_y = keras.utils.to_categorical(raw.label, num_classes)  # One-Hot Encoding

    num_images = raw.shape[0]
    x_as_array = raw.values[:,1:]  # Image pixels content
    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)  # 4D-array
                                                                            # ,1 : only grey scales images
    out_x = x_shaped_array / 255   # Pixel intensity between 0 and 1: improves the optimisation of the adam
                                   # optimizer with the default parameters
    return out_x, out_y

def data_prep_test(raw):
    num_images = raw.shape[0]
    x_as_array = raw.values[:,:]  # Image pixels content
    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)  # 3D-array
    out_x = x_shaped_array / 255   # Pixel intensity between 0 and 1: improves the optimisation of the adam
                                   # optimizer with the default parameters
    return out_x


def plt_image(prep_img):
    img=np.empty((img_rows, img_cols, 3))
    for i in range(img_rows):
        for j in range(img_cols):
            img[i][j][0]=1-prep_img[i][j][0]
            img[i][j][1]=1-prep_img[i][j][0]
            img[i][j][2]=1-prep_img[i][j][0]
    return(img)

def decode_pred(pred, n_print=2):
    proba=-np.sort(-pred)[0:n_print]
    digit=[]
    for p in proba:
        for i in range(10):
            if (pred[i]==p): digit.append(i)
    return(proba, digit)


########## MAIN ##########

if (TRAIN):
    ## Load training data
    train_file = "./input/digit_recognizer/train.csv"
    raw_data = pd.read_csv(train_file)
    x, y = data_prep(raw_data)


    ## Specify model
    model = Sequential()
    # Input layer
    model.add(Conv2D(20, kernel_size=(3, 3), activation='relu', input_shape=(img_rows, img_cols, 1)))
                #   (number of convolutions/filters, size of the conv, activation function
    # Hidden layers
    model.add(Dropout(0.3))
    model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))
    model.add(Dropout(0.3))
    #model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))
    model.add(Flatten())                       # Flatten layer
                                               # Convert the output of the previous layers into a 1D representation
    model.add(Dense(128, activation='relu'))   # Dense layer with 128 nodes
                                               # Perform usually better when adding a dense layer in between
                                               # the flatten layer and the final layer
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))


    ## Compile model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])
    
    ## Fit model
    model.fit(x, y, batch_size=128, epochs=2, validation_split = 0.2)

    ## Save model
    save_model(model, "./input/digit_recognizer/model.h5")


if (PREDICT):
    model = load_model("./input/digit_recognizer/model.h5")

    ## Load test data
    test_file = "./input/digit_recognizer/test.csv"
    raw_data = pd.read_csv(test_file)
    x = data_prep_test(raw_data)

    #print(x[0])
    #plt.imshow(plt_image(x[0]))
    #plt.show()
    

    n_img = 40
    plt.figure()
    for i in range(n_img):
        proba, digit = decode_pred(model.predict(x[i:i+1,:,:,:])[0],2)
        #print("Image {}: \t{}\t{:2.2f} \n \t\t{}\t{:2.2f}".format(i+1,digit[0],proba[0],digit[1],proba[1]))
        print("Image {}: {}  {:2.2f} \n        {} {}  {:2.2f}".format(i+1,digit[0],proba[0],int(np.log10(i+1))*" ",digit[1],proba[1]))
        
        plt.subplot(4, n_img/4, i+1)
        plt.imshow(plt_image(x[i]))
    plt.show()


