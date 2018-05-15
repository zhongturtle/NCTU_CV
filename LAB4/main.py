from __future__ import print_function
from random import shuffle
import os
import argparse
import pickle
from glob import glob
import cyvlfeat 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import scipy.spatial.distance as distance
import operator
import pdb
import numpy as np
from time import time
import pdb
from cyvlfeat.sift.dsift import dsift
from cyvlfeat.kmeans import kmeans
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

def nearest_neighbor(train_image_feats, train_labels, test_image_feats):
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
 
def build_vocabulary(image_paths, vocab_size):

    feature_bag = []
    
    print("SIFT features extracting")
    for image_path in image_paths:
        image = np.asarray(Image.open(image_path),dtype='float32')
        frames, descriptors = dsift(image, step=[5,5], fast=True)
        feature_bag.append(descriptors)
    feature_bag = np.concatenate(feature_bag, axis=0).astype('float32')
    print("Computing vocabulary")
    vocabulary = kmeans(feature_bag, vocab_size, initialization="PLUSPLUS")        

    return vocabulary

def get_bags_of_sifts(image_paths):

    with open('vocab.pkl', 'rb') as vocab:
        vocabulary = pickle.load(vocab)
        image_feature = np.zeros((len(image_paths),len(vocabulary)))
        
    for i, path in enumerate(image_paths):
        
        image = np.asarray(Image.open(path), dtype = 'float32')
        frames, descriptors = dsift(image, step=[9,9], fast=True)
        
        dist = distance.cdist(vocabulary, descriptors, 'euclidean')
        mdist = np.argmin(dist, axis = 0)
        histo, bins = np.histogram(mdist, range(len(vocabulary)+1))
        if np.linalg.norm(histo) != 0:
            image_feature[i, :] = histo / np.linalg.norm(histo)        
        elif np.linalg.norm(histo) == 0:
            image_feature[i, :] = histo
        else:
            print("something wrong, check the np")
            
    return image_feature

DATA_PATH = 'data/'

CATEGORIES = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office', 'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street', 'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']

CATE2ID = {v: k for k, v in enumerate(CATEGORIES)}

ABBR_CATEGORIES = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office', 'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street', 'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']

#FEATURE  = 'tiny_image'
FEATURE = 'bag_of_sift'
CLASSIFIER = 'nearest_neighbor'

NUM_TRAIN_PER_CAT = 100


def main():

    print("Get the data set")
    train_image, test_image, train_labels, test_labels = get_data(DATA_PATH, CATEGORIES, NUM_TRAIN_PER_CAT)
    if FEATURE  == 'tiny_image':
        print("Make tiny image training and testing dada feature")
        train_image_feats = make_tiny_image(train_image)
        test_image_feats = make_tiny_image(test_image)
    elif FEATURE == 'bag_of_sift':
        print("Make bag_of_sift training and testing dada feature")
        
        if os.path.isfile('vocab.pkl') is True:
            print("reuse the \" vocab.pkl\"") 
        #it take about 40 min to compute, i upload the result, do not compute again XD 
        
        elif os.path.isfile('vocab.pkl') is False:
            print('Can not fount vocub.pkl !!! Make one from dataset\n')
            start_time = time()
            vocabulary_size = 400
            vocabulary = build_vocabulary(train_image, vocab_size)
            with open('vocab.pkl', 'wb') as handle:
                pickle.dump(vocabulary, handle, protocol=pickle.HIGHEST_PROTOCOL)
            finish_time = time()
            print('it take ', finish_time - start_time , "sec to finish computing")

        train_image_feats = get_bags_of_sifts(train_image);
        with open('train_image_feats.pkl', 'wb') as handle:
            pickle.dump(train_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)

        test_image_feats  = get_bags_of_sifts(test_image);
        with open('test_image_feats.pkl', 'wb') as handle:
            pickle.dump(test_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)



    if CLASSIFIER == 'nearest_neighbor':
        print("Training....")
        predicted_categories = nearest_neighbor(train_image_feats, train_labels, test_image_feats)
    
    
    
    
    
    
    
    print("Testing....")
    accuracy = float(len([x for x in zip(test_labels,predicted_categories) if x[0]== x[1]]))/float(len(test_labels))
    print("Accuracy = ", accuracy)
    test_labels_ids = [CATE2ID[x] for x in test_labels]
    predicted_categories_ids = [CATE2ID[x] for x in predicted_categories]
    train_labels_ids = [CATE2ID[x] for x in train_labels]
    
   
    build_confusion_mtx(test_labels_ids, predicted_categories_ids, ABBR_CATEGORIES)

if __name__ == '__main__':
    main()

