from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.engine import Input
import numpy as np
import glob
import os
import cv2
import dicom
import pandas as pd
from sklearn import cross_validation

from matplotlib import pyplot as plt


def get_3d_data(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    return np.stack([s.pixel_array for s in slices])

def get_data_id(path):
    sample_image = get_3d_data(path)
    sample_image[sample_image == -2000] = 0
    print('sample',sample_image.shape)
    f, plots = plt.subplots(4, 5, sharex='col', sharey='row', figsize=(10, 8))

    batch = []
    cnt = 0
    dx = 40
    ds = 512
    for i in range(0, sample_image.shape[0] - 3, 3):
        tmp = []
        for j in range(3):
            img = sample_image[i + j]
            img = 255.0 / np.amax(img) * img
            img = cv2.equalizeHist(img.astype(np.uint8))
            img = img[dx: ds - dx, dx: ds - dx]
            img = cv2.resize(img, (224, 224))
            tmp.append(img)

        tmp = np.array(tmp)
        print('tmp',tmp.shape)
#        tmp_conv=np.swapaxes(np.swapaxes(tmp, 1, 2), 0, 1)
        tmp_conv = Input(shape=(224,224,3))
        batch.append(np.array(tmp_conv))

        if cnt < 20:
            plots[cnt // 5, cnt % 5].axis('off')
            plots[cnt // 5, cnt % 5].imshow(np.swapaxes(tmp, 0, 2))
        cnt += 1
    plt.show()
    batch = np.array(batch)
    return batch

def train_xgboost():
    df = pd.read_csv('./stage1_labels.csv')
    print(df.head())

    x = np.array([np.mean(np.load('stage1/%s.npy' % str(id)), axis=0) for id in df['id'].tolist()])
    y = df['cancer'].as_matrix()

    trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(x, y, random_state=42, stratify=y,
                                                                   test_size=0.20)

    clf = xgb.XGBRegressor(max_depth=10,
                           n_estimators=1500,
                           min_child_weight=9,
                           learning_rate=0.05,
                           nthread=8,
                           subsample=0.80,
                           colsample_bytree=0.80,
                           seed=4242)

    clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='logloss', early_stopping_rounds=50)
    return clf


model = ResNet50(weights='imagenet')

#img_path = 'elephant.jpg'
#img = image.load_img(img_path, target_size=(224, 224))
for folder in glob.glob('./input/sample_images/*'):
    batch = get_data_id(folder)
#    x = image.img_to_array(img)
#    x = np.expand_dims(x, axis=0)
#    batch = preprocess_input(batch)
    preds = model.predict(batch)
    print(feats.shape)
    np.save(folder, feats)

#clf = train_xgboost()
#
#df = pd.read_csv('data/stage1_sample_submission.csv')
#
#x = np.array([np.mean(np.load('./input/sample_images/%s.npy' % str(id)), axis=0) for id in df['id'].tolist()])
#
#pred = clf.predict(x)
#
#df['cancer'] = pred
#df.to_csv('subm1.csv', index=False)
#print(df.head())