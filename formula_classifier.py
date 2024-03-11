from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import pickle
import cv2
import os
import threading

def classify(args):
    # load the trained convolutional neural network and the multi-label
    # binarizer
    print("[INFO] loading network...")
    model = load_model(args.model)
    model.summary()
    mlb = pickle.loads(open(args.labelbin, "rb").read())

    # load the images
    file = open(args.image, 'rb')
    data = pickle.load(file)
    file.close()
    formulas = data[0]
    sequences = data[1]
    
    # Extracting the specified equation number
    formula_number = args.equation_number
    if formula_number >= len(formulas):
        print(f"Equation number {formula_number} does not exist.")
        return

    formula = formulas[formula_number]
    sequence = sequences[formula_number]['sequence']

    prediction = ""
    for i in range(formula['length']):
        print(formula['images'][i])
        # Decode the formula images
        retval, buffer = cv2.imencode('.jpg', formula['images'][i]) # Convert numpy array to bytes
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR) # Convert bytes to numpy array
        output = imutils.resize(image, width=400)

        # pre-process the image for classification
        image = cv2.resize(image, (45, 45))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # classify the input image then find the indexes of the two class-
        # labels with the *largest* probability
        print("[INFO] classifying image...")
        proba = model.predict(image)[0]
        idxs = np.argsort(proba)[::-1][:2]

        # loop over the indexes of the high confidence class labels
        for (j, idx) in enumerate(idxs):
            # build the label and draw the label on the image
            label = "{}: {:.2f}%".format(mlb.classes_[idx], proba[idx] * 100)
            cv2.putText(output, label, (10, (j * 30) + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # show the probabilities for each of the individual labels
        for (label, p) in zip(mlb.classes_, proba):
            print("{}: {:.2f}%".format(label, p * 100))

        # show the output image if specified
        if args.show_output:
            cv2.imshow("Output", output)
            cv2.waitKey(0)

        prediction += mlb.classes_[proba.argmax(axis=-1)]

    # Print the predicted formula
    print("The predicted formula is: ", prediction)


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
    parser.add_argument("-n", "--equation_number", type=int, default=1,
        help="number of the equation to predict (default: 1)")
    parser.add_argument("-s", "--show_output", action="store_true", default=True,
        help="show output images")
    args = parser.parse_args()

    classify(args)
