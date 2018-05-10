from __future__ import print_function
from random import shuffle
import os
import argparse
import pickle
from glob import glob

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import scipy.spatial.distance as distance
import operator
import pdb
import numpy as np
def get_data(data_path, categories, num_train_per_cat):
    num_categories = len(categories)

    train_image = []
    test_image = []

    train_labels = []
    test_labels = []

    for category in categories:

        train_paths = glob(os.path.join(data_path, 'train', category, '*.jpg'))
        for i in range(num_train_per_cat):
            train_image.append(train_paths[i])
            train_labels.append(category)

        test_paths = glob(os.path.join(data_path, 'test', category, '*.jpg'))
        for i in range(num_train_per_cat):
            test_image.append(test_paths[i])
            test_labels.append(category)

    return train_image, test_image, train_labels, test_labels

def make_tiny_image(image_paths):

    tiny_images = np.zeros((len(image_paths), 16*16))
    
    for i, image_data in enumerate(image_paths):
        
        image = Image.open(image_data)
        image_re = np.asarray(image.resize((16,16), Image.ANTIALIAS), dtype = 'float32').flatten()
        image_nm = (image_re - np.mean(image_re))/np.std(image_re)
        tiny_images[i,:] = image_nm
        
    return tiny_images

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats):
    k = 11
    test_predicts = []
    dist = distance.cdist(train_image_feats, test_image_feats, 'euclidean')
    
    for i in range(dist.shape[1]):
        ans = np.argsort(dist[:,i])
        nn = dict()
        for j in range(k):
            if train_labels[ans[j]] in nn.keys():
                nn[train_labels[ans[j]]] += 1
            else :
                nn[train_labels[ans[j]]] = 1
 
        snn = sorted(nn.items(), key = operator.itemgetter(1), reverse=True)
        test_predicts.append(snn[0][0])

    return test_predicts

DATA_PATH = 'data/'

CATEGORIES = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
              'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
              'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']

CATE2ID = {v: k for k, v in enumerate(CATEGORIES)}

ABBR_CATEGORIES = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office', 'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street', 'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']

FEATURE  = 'tiny_image'

CLASSIFIER = 'nearest_neighbor'

NUM_TRAIN_PER_CAT = 100

def build_confusion_mtx(test_labels_ids, predicted_categories, abbr_categories):

    cm = confusion_matrix(test_labels_ids, predicted_categories)
    np.set_printoptions(precision=2)

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure()
    plot_confusion_matrix(cm_normalized, abbr_categories, title='Normalized confusion matrix')

    plt.show()
     
def plot_confusion_matrix(cm, category, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(category))
    plt.xticks(tick_marks, category, rotation=75)
    plt.yticks(tick_marks, category)
    plt.tight_layout()
    plt.ylabel('Ground_Truth label')
    plt.xlabel('Predicted label')
 

def main():

    print("Get the data set")
    train_image, test_image, train_labels, test_labels = get_data(DATA_PATH, CATEGORIES, NUM_TRAIN_PER_CAT)
    print("Make training and testing dada feature")
    train_image_feats = make_tiny_image(train_image)
    test_image_feats = make_tiny_image(test_image)

    print("Training....")
    predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats)
    
    print("Testing....")
    accuracy = float(len([x for x in zip(test_labels,predicted_categories) if x[0]== x[1]]))/float(len(test_labels))
    print("Accuracy = ", accuracy)
    test_labels_ids = [CATE2ID[x] for x in test_labels]
    predicted_categories_ids = [CATE2ID[x] for x in predicted_categories]
    train_labels_ids = [CATE2ID[x] for x in train_labels]
    
   
    build_confusion_mtx(test_labels_ids, predicted_categories_ids, ABBR_CATEGORIES)

if __name__ == '__main__':
    main()

