import os
import sys
import glob

import numpy as np
import pandas as pd

import SimpleITK as sitk
from scipy.ndimage import rotate, imread
from PIL import Image
from scipy.misc import imread
from sklearn.cross_validation import train_test_split

import tensorflow as tf
import tflearn
from tflearn.data_utils import build_hdf5_image_dataset
import h5py

from joblib import Parallel, delayed

def do_test_train_split(filename):
    """
    Does a test train split if not previously done
    """
    candidates = pd.read_csv(filename)

    positives = candidates[candidates['class']==1].index  
    negatives = candidates[candidates['class']==0].index

    ## Under Sample Negative Indexes
    np.random.seed(900)
    negIndexes = np.random.choice(negatives, len(positives)*2, replace=False)

    candidatesDf = candidates.iloc[list(positives)+list(negIndexes)]

    X = candidatesDf.iloc[:,:-1]
    y = candidatesDf.iloc[:,-1]
    y.to_pickle('./truthdict.pkl')
    print y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.20, random_state = 42)
    
    X_train.to_pickle('./traindata')
    y_train.to_pickle('./trainlabels')
    X_test.to_pickle('./testdata')
    y_test.to_pickle('./testlabels')
    X_val.to_pickle('./valdata')
    y_val.to_pickle('./vallabels')
    print("lol")

if __name__ == '__main__':
    print("lol")
    do_test_train_split('/Users/weichaozhou/Documents/Boston University/EC500/DSB3Tutorial/luna2016/candidates.csv')
