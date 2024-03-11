import os
import pickle
import numpy as np
# To create the neural network 
from keras.models import Sequential
# To add hidden layers to the neural network
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.optimizers import Adam, RMSprop
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.backend import image_data_format

NUMERICAL_SYMBOLS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
MATH_SYMBOLS = ['+', '=']
SYMBOLS = NUMERICAL_SYMBOLS + MATH_SYMBOLS

def create_LeNet_model(height, width, depth, classes):
    # Parameters
    input_shape = (height, width, depth)

    # Create model
    print("Creating Model...")
    model = Sequential()
    model.add(Conv2D(30, 5, input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(15, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(13, activation='softmax'))

    return model

def compile_LeNet_model(model):
    # Compile the model
    print("Compiling Model...")
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model

def create_vgg_model(width, height, depth, classes):
    # Parameters
    input_shape = (height, width, depth)

    # Create model
    print("Creating Model...")
    model = Sequential()
    model.add(Conv2D(32, (2, 2), input_shape = input_shape)) 
    model.add(Activation('relu')) 
    model.add(MaxPooling2D(pool_size =(2, 2))) 
    
    model.add(Conv2D(32, (2, 2))) 
    model.add(Activation('relu')) 
    model.add(MaxPooling2D(pool_size =(2, 2))) 
    
    model.add(Conv2D(64, (2, 2))) 
    model.add(Activation('relu')) 
    model.add(MaxPooling2D(pool_size =(2, 2))) 
    
    model.add(Flatten()) 
    model.add(Dense(64)) 
    model.add(Activation('relu')) 
    model.add(Dropout(0.5)) 
    model.add(Dense(classes)) 
    model.add(Activation('softmax')) 

    return model

def compile_vgg_model(model):
    # Compile the model 
    print("Compiling Model...")
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model

def train_LeNet(args):
    pass

def train_vgg(args):
    #Initialize parametres
    EPOCHS = 12
    BS = 100 #Batch size
    LR = 1e-3 #Learning rate 0.001
    img_dim = (45,45,3)
    train_data_dir = 'splited_dataset/train'
    test_data_dir = 'splited_dataset/test'
    #Nbr of training images
    train_samples_nbr  = sum(len(files) for _, _, files in os.walk(f'splited_dataset/train'))
    #Nbr of testing images
    test_samples_nbr  = sum(len(files) for _, _, files in os.walk(f'splited_dataset/test'))

    print(f"Training samples: {train_samples_nbr}")
    print(f"Testing samples: {test_samples_nbr}")

    # Info about our train Dataset
    nbr_of_pictures = []
    labels = []
    for root, dirs, files in os.walk(train_data_dir):
        # Exclude the top directory itself
        print(root)
        if root != train_data_dir:
            labels.append(os.path.basename(root))
            nbr_of_pictures.append(len(files))

    nbr_labels = len(labels)
    
    print(f"Labels: {nbr_labels}")
    print(f"Labels: {labels}")
    print(f"Number of pictures: {nbr_of_pictures}")

    if image_data_format() == 'channels_first':
        input_shape = (img_dim[2], img_dim[0], img_dim[1])
    else:
        input_shape = (img_dim[0], img_dim[1], img_dim[2])

    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(labels)
    
    model = create_vgg_model(width=img_dim[0], height=img_dim[1], depth=img_dim[2], classes=nbr_labels)

    # Compile the model
    opt = Adam(learning_rate=LR)
    model.compile(loss="categorical_crossentropy", 
                  optimizer=opt, 
                  metrics=["accuracy"])

    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.0,
        zoom_range=0.0,
        featurewise_center=False,# set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0.0,  # randomly rotate images in the range (deg 0 to 180)
        width_shift_range=0.0,  # randomly shift images horizontally
        height_shift_range=0.0,  # randomly shift images vertically
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False
        )

    # data augmentation for testing
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Training data generator
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_dim[0], img_dim[1]),
        batch_size=BS,
        class_mode='categorical')

    # Testing data generator
    validation_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_dim[0], img_dim[1]),
        batch_size=BS,
        class_mode='categorical')

    # Training
    history = model.fit(
        train_generator,
        steps_per_epoch=train_samples_nbr // BS,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=test_samples_nbr // BS)

    # save the model to disk
    print("Saving the model...")
    #model.save(args["model"])
    model.save("model.keras")
    model.save_weights("model.weights.h5")
    #save the multi-label binarizer to disk
    print("Saving Labels...")
    # f = open(args["labelbin"], "wb")
    f = open("labels.pickle", "wb")
    f.write(pickle.dumps(mlb))
    f.close()

    # Evaluating the model / Get Validation accuracy on sample from validation set 
    scores = model.evaluate(validation_generator, steps=test_samples_nbr//BS, verbose=1)

    print("Accuracy = ", scores[1])


if __name__ == "__main__":
    import argparse

    # construct the argument parse and parse the arguments (for command line)
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", # --dataset : The path to our dataset. add required=True if you want
        help="path to input dataset (i.e., directory of images)")
    parser.add_argument("-m", "--model", # --model : The path to our output serialized Keras model.
        help="path to output model")
    parser.add_argument("-l", "--labelbin", # --labelbin : The path to our output multi-label binarizer object.
        help="path to output label binarizer")
    parser.add_argument("-p", "--plot", type=str, default="plot.png", # --plot : The path to our output plot of training loss and accuracy.
        help="path to output accuracy/loss plot")
    args = parser.parse_args()

    train_vgg(args)