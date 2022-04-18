import pandas as pd
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import numpy as np
import sklearn.preprocessing as preprocessing
import os
import cv2

# Note that you can save models in different formats. Some format needs to save/load model and weight separately.
# Some saves the whole thing together. So, for your set up you might need to save and load differently.

def load_model_weights(model, weights = None):
    my_model = tf.keras.models.load_model(model, custom_objects={'KerasLayer':hub.KerasLayer})
    my_model.summary()
    return my_model

def get_images_labels(df, classes, img_height, img_width):
    test_images = []
    test_labels = np.array([])
    # Write the code as needed for your code
    for index, row in df.iterrows():
        label = row[1]
        img = tf.io.read_file(row[0])
        img = decode_img(img, img_height, img_width)
        test_images.append(img)
        test_labels = np.append(test_labels, label)
    return test_images, test_labels

def decode_img(img, img_height, img_width):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize(img, [img_height, img_width])

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Trasnfer Learning Test")
    parser.add_argument('--model', type=str, default='my_model.h5', help='Saved model')
    parser.add_argument('--weights', type=str, default=None, help='weight file if needed')
    parser.add_argument('--test_csv', type=str, default='flowers_test.csv', help='CSV file with true labels')

    args = parser.parse_args()
    model = args.model
    weights = args.weights
    test_csv = args.test_csv

    test_df = pd.read_csv(test_csv)
    #classes = {'astilbe', 'bellflower', 'black-eyed susan', 'calendula', 'california poppy','carnation', 'common daisy', 'coreopsis', 'dandelion', 'iris', 'rose', 'sunflower', 'tulip'}
    classes = ['astilbe', 'bellflower', 'black-eyed susan', 'calendula', 'california poppy','carnation', 'common daisy', 'coreopsis', 'dandelion', 'iris', 'rose', 'sunflower', 'tulip']

    # Rewrite the code to match with your setup
    test_images, test_labels = get_images_labels(test_df, classes, 224, 224)

    # one hot encoding of the 13 flowers classes
    test_images = np.array(test_images)
    test_images = test_images.astype('float32')
    targets = np.array(classes)
    labelEnc = preprocessing.LabelEncoder()
    new_target = labelEnc.fit_transform(targets)
    onehotEnc = preprocessing.OneHotEncoder()
    onehotEnc.fit(new_target.reshape(-1, 1))
    targets_trans = onehotEnc.transform(new_target.reshape(-1, 1))
    labels_enc = targets_trans.toarray()
    
    ts = test_labels
    test_labels = []

    # converting the test labels from strings to their one hot encoded representations
    for s in ts:
        s = s.strip()
        if s == "astilbe":
            test_labels.append(labels_enc[0])
        if s == "bellflower":
            test_labels.append(labels_enc[1])
        if s == "black-eyed susan":
            test_labels.append(labels_enc[2])
        if s == "calendula":
            test_labels.append(labels_enc[3])
        if s == "california poppy":
            test_labels.append(labels_enc[4])
        if s == "carnation":
            test_labels.append(labels_enc[5])
        if s == "common daisy":
            test_labels.append(labels_enc[6])
        if s == "coreopsis":
            test_labels.append(labels_enc[7])
        if s == "dandelion":
            test_labels.append(labels_enc[8])
        if s == "iris":
            test_labels.append(labels_enc[9])
        if s == "rose":
            test_labels.append(labels_enc[10])
        if s == "sunflower":
            test_labels.append(labels_enc[11])
        if s == "tulip":
            test_labels.append(labels_enc[12])
        
    test_labels = np.array(test_labels)

    # loading in the models
    my_model = load_model_weights(model)

    # code to perform normalizaiton by dividing the array values by 255
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    preprocessing_model = tf.keras.Sequential([normalization_layer])
    
    # batching the dataset to match the neural net input shape
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images,test_labels)).batch(1)
    # mapping the images to their labels and performing normalization
    test_dataset = test_dataset.map(lambda images, labels:
                        (preprocessing_model(images), labels))
    print(test_dataset)
    loss, acc = my_model.evaluate(test_dataset)
    
    # loss, acc = my_model.evaluate(test_images, test_labels, verbose=2)
    print('Test model, accuracy: {:5.5f}%'.format(100 * acc))

    