import os
import sys
import cv2 # load images and processing
import numpy as np # working with numpy arrays
import matplotlib.pyplot as plt # visualization of the digits
import tensorflow as tf # machine learning
var = 0 # use for for training/test ou uncomment the other if clauses to use command line args
if var == 0: # sys.argv[1] == 'train': # train
    tf.keras.backend.clear_session() # clears previous models. Keras starts with a blank state at each iteration
                                     # and memory consumption is constant over time.
    mnist = tf.keras.datasets.mnist # skips the need to download csv files and its processing
                                    # spliting into training data and testing data 
                                    # the MNIST database contains 60,000 training images and 10,000 testing images. 
                                    # roughly an 85/15 split

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x -> pixel data, y -> classification
    # normalize the pixels as to have values from 0 to 1 (easier for the NN to calculate)
    
    train = tf.keras.utils.normalize(x_train, axis=1)
    test = tf.keras.utils.normalize(x_train, axis=1)
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) 
    #instead of a 28x28 grid, the image becames a line 784 pixels
   
    model.add(tf.keras.layers.Dense(128, activation='relu')) # 0 if negative and linear otherwise
    model.add(tf.keras.layers.Dense(256, activation='relu')) # 0 if negative and linear otherwise
    model.add(tf.keras.layers.Dense(10, activation='softmax')) # all 10 neurons add up to 1 (confidance)
    # an obvious 2 will have 0.95 on the third neuron, for example, and low confidance on the other 9 neurons
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Softmax is the only activation function recommended to use with the categorical crossentropy loss function.
    # as the categorical crossentropy loss function relies on the logarithm of every output
    
    model.fit(x_train, y_train, epochs=4)
    # High accuracy and low loss values desirable
    loss, accuracy = model.evaluate(x_test, y_test)
    print(loss) # 0.1816355586051941 not that great
    print(accuracy) # 0.9535999894142151 meh

    model.save('HWModel') 

if var == 1: # sys.argv[1] == 'test': #load
    model = tf.keras.models.load_model('HWModel')

img_n = 1
while os.path.exists(f"Digits/d{img_n}.png"): # using f-strings
    try:
        img = cv2.imread(f"Digits/d{img_n}.png")[:,:,0]   
        # because the image comes in white on black instead of black on white we need to invert evry value
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"my guess is: {np.argmax(prediction)}") #the neuron with the highest number will be selected
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("error")
    finally:
        img_n += 1
