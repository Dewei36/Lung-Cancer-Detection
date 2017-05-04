import numpy as np
import pickle
import scipy as sp
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier as RF
import xgboost as xgb
from skimage.measure import label, regionprops, perimeter
from glob import *

def getRegionFromMap(slice_npy):
    thr = np.where(slice_npy > np.mean(slice_npy),0.,1.0)
    label_image = label(thr)
    labels = label_image.astype(int)
    regions = regionprops(labels)
    return regions

def getRegionMetricRow(filename = "nodules.npy"):
    segment = np.load(filename)
    slices = segment.shape[0]

    all_Area = 0.
    average_Area = 0.
    max_Area = 0.
    average_Ecc = 0.
    average_eqDiameter = 0.
    std_eqDiameter = 0.
    w_X = 0.
    w_Y = 0.
    num_of_Nodes = 0.
    num_of_n_per_slice = 0.
    max_A = 0.10 * 512 * 512

    areas = []
    eD = []
    for s in range(slices):
        regions = getRegionFromMap(segment[s,:,:])
        for r in regions:
            if r.area > max_A:
                continue
            all_Area += r.area
            areas.append(r.area)
            average_Ecc += r.eccentricity
            average_eqDiameter += r.equivalent_diameter
            eD.append(r.equivalent_diameter)
            w_X += r.centroid[0]*r.area
            w_Y += r.centroid[1]*r.area
            num_of_Nodes += 1
    if all_Area==0:
        return np.zeros(9)
    w_X = w_X / all_Area
    w_Y = w_Y / all_Area
    average_Area = all_Area / num_of_Nodes
    average_Ecc = average_Ecc / num_of_Nodes
    average_eqDiameter = average_eqDiameter / num_of_Nodes
    std_eqDiameter = np.std(eD)

    max_Area = max(areas)


    num_of_n_per_slice = num_of_Nodes*1. / slices


    return np.array([average_Area,max_Area,average_Ecc,average_eqDiameter,\
                     std_eqDiameter, w_X, w_Y, num_of_Nodes, num_of_n_per_slice])


def createFeatureDataset(nodules_files=None):
    if nodules_files == None:

        noddir = "/Users/weichaozhou/Documents/Boston University/EC500/DSB3Tutorial/luna2016/"
        nodules_files = glob(noddir +"masks*.npy")
    truth_d = pickle.load(open("truthdict.pkl",'r'))
    num_of_features = 9
    features = np.zeros((len(nodules_files),num_of_features))
    truth = np.zeros((len(nodules_files)))

    for i,f in enumerate(nodules_files):
        patient_ID = f.replace(".","_").split("_")[2]
        try:
            truth[i] = truth_d[int(patient_ID)]
        except KeyError:
            truth[i] =0
        print f
        features[i] = getRegionMetricRow(f)

    np.save("dataY.npy", truth)
    np.save("dataX.npy", features)


def logloss(act, predicted):
    predicted = sp.minimum(1-(1e-15), sp.maximum(1e-15, predicted))
    v1 = act*sp.log(predicted)
    v2 = sp.subtract(1,act)
    v3 = sp.log(sp.subtract(1,predicted))
    LogLoss = sum(v1 + v2 * v3)
    LogLoss = LogLoss * (-1.0/len(act))
    return LogLoss


def classifyData():
    X = np.load("dataX.npy")
    Y = np.load("dataY.npy")

    k_fold = KFold(Y, n_folds=3)
    y_pred = Y * 0
    for train, test in k_fold:
        X_train, y_train, X_test, y_test = X[train,:], Y[train], X[test,:], Y[test]
        clf = RF(n_estimators=100, n_jobs=3)
        clf.fit(X_train, y_train)
        y_pred[test] = clf.predict(X_test)
    print(classification_report(Y, y_pred, target_names=["No Cancer", "Cancer"]))
    print("logloss = ",logloss(Y, y_pred))

    # All Cancer
    print("prediction of the positive")
    y_pred = np.ones(Y.shape)
    print(classification_report(Y, y_pred, target_names=["No Cancer", "Cancer"]))
    print("logloss = ",logloss(Y, y_pred))

    # No Cancer
    print("prediction of the negative")
    y_pred = Y*0
    print(classification_report(Y, y_pred, target_names=["No Cancer", "Cancer"]))
    print("logloss = ",logloss(Y, y_pred))

    # try XGBoost
    print("XGBoost")
    k_fold = KFold(Y, n_folds=3)
    y_pred = Y * 0
    for train, test in k_fold:
        X_train, y_train, X_test, y_test = X[train,:], Y[train], X[test,:], Y[test]
        clf = xgb.XGBClassifier(objective="binary:logistic")
        clf.fit(X_train, y_train)
        y_pred[test] = clf.predict(X_test)
    print(classification_report(Y, y_pred, target_names=["No Cancer", "Cancer"]))
    print("logloss = ",logloss(Y, y_pred))

if __name__ == "__main__":
    from sys import argv

    createFeatureDataset()
    classifyData()
