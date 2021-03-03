#!/usr/bin/python3


######################## DEEP LEARNING TUTORIAL ########################

from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.applications import ResNet50
from tensorflow.keras.applications import VGG16
from utils.decode_predictions import decode_predictions


image_size = 224
def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    output = preprocess_input(img_array)
    return(output)


run_dog     = 0
run_hot_dog = 1

### DOG EXAMPLE

if (run_dog):
    image_dir = './input/dog_breed_identification/train/'
    img_paths = [join(image_dir, filename) for filename in 
                               ['0246f44bb123ce3f91c939861eb97fb7.jpg',
                                '84728e78632c0910a69d33f82e62638c.jpg',
                                '8825e914555803f4c67b26593c9d5aff.jpg',
                                '91a5e8db15bccfb6cfa2df5e8b95ec03.jpg']]
                                
                                
    model = ResNet50(weights='./input/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
    test_data = read_and_prep_images(img_paths)
    preds = model.predict(test_data)
    
    #print(preds)
    #print("\n")
    #print(preds[0])
    
    most_likely_labels = decode_predictions(preds, top=3, class_list_path='./input/imagenet_class_index.json')
    for i, img_path in enumerate(img_paths):
        print("Most likely labels:")
        print(most_likely_labels[i])
        plt.figure()
        img=mpimg.imread(img_path)
        plt.imshow(img) 
        plt.show()  # display it

### HOT-DOG EXAMPLE

if (run_hot_dog):
    def is_hot_dog(preds):
        decoded = decode_predictions(preds, top=1, class_list_path='./input/imagenet_class_index.json')
        L=[decoded[i][0][1]=='hotdog' for i in range(len(decoded))]
        return(L)
    
    def calc_accuracy(model, paths_to_hotdog_images, paths_to_other_images):
        hot_dog_data = read_and_prep_images(paths_to_hotdog_images)
        preds_hot_dog = model.predict(hot_dog_data)
        L1 = is_hot_dog(preds_hot_dog)
        not_hot_dog_data = read_and_prep_images(paths_to_other_images)
        preds_not_hot_dog = model.predict(not_hot_dog_data)
        L2 = is_hot_dog(preds_not_hot_dog)
        f1=0
        f2=0
        for i in range (len(L1)):
            if (L1[i]==True):
                f1+=1
        print("True hot dog found: {}".format(f1))
        for i in range (len(L2)):
            if (L2[i]==False):
                f2+=1
        print("True not hot dog found: {}".format(f2))
        print("Number of pictures read: {}".format(len(L1)+len(L2)))
        return (float(f1+f2)/float(len(L1)+len(L2)))    
    
    hot_dog_image_dir = './input/food/train/hot_dog'
    hot_dog_paths = [join(hot_dog_image_dir,filename) for filename in 
                                ['1000288.jpg',
                                 '127117.jpg']]
    
    not_hot_dog_image_dir = './input/food/train/not_hot_dog'
    not_hot_dog_paths = [join(not_hot_dog_image_dir, filename) for filename in
                                ['823536.jpg',
                                 '99890.jpg']]
    
    img_paths = hot_dog_paths + not_hot_dog_paths
    
    model = ResNet50(weights='./input/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
    test_data = read_and_prep_images(img_paths)
    preds = model.predict(test_data)
    most_likely_labels = decode_predictions(preds, top=3, class_list_path='./input/imagenet_class_index.json')    
    for i, img_path in enumerate(img_paths):
        print("Most likely labels:")
        print(most_likely_labels[i])
        plt.figure()
        img=mpimg.imread(img_path)
        plt.imshow(img) 
        plt.show()  # display it

    print("\n")
    model_accuracy = calc_accuracy(model, hot_dog_paths, not_hot_dog_paths)
    print("Fraction correct in small test set: {}".format(model_accuracy))

    vgg16_model = VGG16(weights='./input/vgg16_weights_tf_dim_ordering_tf_kernels.h5')
    print("\n")
    vgg16_accuracy = calc_accuracy(vgg16_model, hot_dog_paths, not_hot_dog_paths)
    print("MODEL VGG16 - Fraction correct in small dataset: {}".format(vgg16_accuracy))

