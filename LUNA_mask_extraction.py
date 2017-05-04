from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd
import os


def get(files, name):
    for file in files:
        if name in file:
            return(file)

def make_mask(center,diam,z,width,length,spacing,origin):
    mask = np.zeros([length,width])
    center = (center-origin)/spacing
    diam = int(diam/spacing[0]+5)
    xmin = np.max([0,int(center[0]-diam)-5])
    xmax = np.min([width-1,int(center[0]+diam)+5])
    ymin = np.max([0,int(center[1]-diam)-5])
    ymax = np.min([length-1,int(center[1]+diam)+5])
    
    xrange = range(xmin,xmax+1)
    yrange = range(ymin,ymax+1)
    
    x_data = [x*spacing[0]+origin[0] for x in range(width)]
    y_data = [x*spacing[1]+origin[1] for x in range(length)]
    
    for x in xrange:
        for y in yrange:
            p_x = spacing[0]*x + origin[0]
            p_y = spacing[1]*y + origin[1]
            if np.linalg.norm(center-np.array([p_x,p_y,z]))<=diam:
                mask[int((p_y-origin[1])/spacing[1]),int((p_x-origin[0])/spacing[0])] = 1.0
    return(mask)

luna = "/Users/weichaozhou/Documents/Boston University/EC500/DSB3Tutorial/LUNA2016/"
subset = luna+"subset0/"
output = "/Users/weichaozhou/Documents/Boston University/EC500/DSB3Tutorial/LUNA2016/"
files=glob(subset+"*.mhd")
node = pd.read_csv(luna+"annotations.csv")
node["file"] = node["seriesuid"].map(lambda file_name: get(files, file_name))
node = node.dropna()




for index_file, img_file in enumerate(files):
    mini_node = node[node["file"]==img_file]
    if mini_node.shape[0]>0:
        itk_img = sitk.ReadImage(img_file)
        img_array = sitk.GetArrayFromImage(itk_img)
        height, length, width = img_array.shape
        origin = np.array(itk_img.GetOrigin())
        spacing = np.array(itk_img.GetSpacing())
        for node_idx, cur_row in mini_node.iterrows():
            node_x = cur_row["coordX"]
            node_y = cur_row["coordY"]
            node_z = cur_row["coordZ"]
            diam = cur_row["diameter_mm"]
            imgs = np.ndarray([3,length,width],dtype=np.float32)
            masks = np.ndarray([3,length,width],dtype=np.uint8)
            center = np.array([node_x, node_y, node_z])
            center = np.rint((center-origin)/spacing)
            for i, i_z in enumerate(np.arange(int(center[2])-1, int(center[2])+2).clip(0, height-1)):
                mask = make_mask(center, diam, i_z*spacing[2]+origin[2], width, length, spacing, origin)
                masks[i] = mask
                imgs[i] = img_array[i_z]
                np.save(os.path.join(output,"images_%04d_%04d.npy" % (index_file, node_idx)),imgs)
        np.save(os.path.join(output,"masks_%04d_%04d.npy" % (index_file, node_idx)),masks)






