import numpy as np
import os
import cv2
import random
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

CATEGORIES = ["star", "galaxy"]
IMGSIZE = 128


def createTrainingData():
    # Directory of the training dataset
    DATADIR = "C:/Users/HENAFF/Documents/StarGalaxy Classification/data/train/"
    training_data = []
    train_X, train_Y = [], []

    # Going through different category of data
    for cat in CATEGORIES:
        # Selecting folder betwenen categories
        path = os.path.join(DATADIR, cat)
        # getting index of each category
        classnum = CATEGORIES.index(cat)
        # going through each image
        for img in os.listdir(path):
            try:
                # Reading and giving a gray scale
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                # resize to the size given
                new_array = cv2.resize(img_array, (IMGSIZE, IMGSIZE))
                training_data.append([new_array, classnum])
            except Exception as e:
                print("Image {} from {} is not working properly".format(img, cat))

    # Shuffle data so the first half is not the first category
    random.shuffle(training_data)

    # split the image into train_X and the classification into train_Y
    for x, y in training_data:
        train_X.append(x)
        train_Y.append(y)

    # change to numpy array
    train_X = np.array(train_X)
    train_Y = np.array(train_Y)

    # Useless but classy
    training_data.clear()

    return train_X, train_Y


def createTestingData():
    # Directory of the testing dataset
    DATADIR = "C:/Users/HENAFF/Documents/StarGalaxy Classification/data/train/"
    testing_data = []
    test_X, test_Y = [], []

    # Going through different category of data
    for cat in CATEGORIES:
        # Selecting folder betwenen categories
        path = os.path.join(DATADIR, cat)
        # getting index of each category
        classnum = CATEGORIES.index(cat)
        # going through each image
        for img in os.listdir(path):
            try:
                # Reading and giving a gray scale
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                # resize to the size given
                new_array = cv2.resize(img_array, (IMGSIZE, IMGSIZE))
                testing_data.append([new_array, classnum])
            except Exception as e:
                print("Image {} from {} is not working properly".format(img, cat))

    # Shuffle data so the first half is not the first category
    random.shuffle(testing_data)

    # split the image into test_X and the classification into test_Y
    for x, y in testing_data:
        test_X.append(x)
        test_Y.append(y)

    # change to numpy array
    test_X = np.array(test_X)
    test_Y = np.array(test_Y)

    # Useless but classy
    testing_data.clear()

    return test_X, test_Y


###################################
# DATA PREPROCESSING
###################################
def dataPreProcessing():
    # Load Data
    train_X, train_Y = createTrainingData()
    test_X, test_Y = createTestingData()

    # Reshape the array into a 3 dimension matrix
    train_X = train_X.reshape(-1, IMGSIZE, IMGSIZE, 1)
    test_X = test_X.reshape(-1, IMGSIZE, IMGSIZE, 1)
    print(train_X.shape, test_X.shape)
    # Convert type of data from int8 to float32 because keras works best with float32
    train_X = train_X.astype('float32')
    test_X = test_X.astype('float32')

    # Change the image to make pixels onto the 0 and 1 interval
    train_X = train_X / 255
    test_X = test_X / 255

    # Convert category to one-hot encoding vector
    train_Y_one_hot = to_categorical(train_Y)
    test_Y_one_hot = to_categorical(test_Y)

    # Split the training data into 80% of training and 20% of validation
    train_X, valid_X, train_label, valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2,
                                                                  random_state=1, shuffle=True)

    return train_X, valid_X, train_label, valid_label, test_X, train_Y_one_hot, test_Y_one_hot
