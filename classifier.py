from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.layers import TFSMLayer
import numpy as np
import imutils
import pickle
import cv2
import os

def classify(args):
    # load the image
    image = cv2.imread(args.image)
    output = imutils.resize(image, width=400)

    # pre-process the image for classification
    image = cv2.resize(image, (45, 45))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # load the trained convolutional neural network and the multi-label
    # binarizer
    print("[INFO] loading network...")
    model = load_model(args.model)
    model.summary()
    mlb = pickle.loads(open(args.labelbin, "rb").read())


    # classify the input image then find the indexes of the two class
    # labels with the *largest* probability
    print("[INFO] classifying image...")
    proba = model.predict(image)[0]
    idxs = np.argsort(proba)[::-1][:2]

    # loop over the indexes of the high confidence class labels
    for (i, j) in enumerate(idxs):
        # build the label and draw the label on the image
        label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
        cv2.putText(output, label, (10, (i * 30) + 25), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # show the probabilities for each of the individual labels
    for (label, p) in zip(mlb.classes_, proba):
        print("{}: {:.2f}%".format(label, p * 100))
    # show the output image
    cv2.imshow("Output", output)
    cv2.waitKey(0)

    print ("This Symbol is :",' '.join(mlb.classes_[proba.argmax(axis=-1)]))

if __name__ == "__main__":
    import argparse

    # construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True,
        help="path to trained model model")
    parser.add_argument("-l", "--labelbin", required=True,
        help="path to label binarizer")
    parser.add_argument("-i", "--image", required=True,
        help="path to input image")
    args = parser.parse_args()

    classify(args)

