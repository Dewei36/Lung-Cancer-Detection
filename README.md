# Lung-Cancer-Detection
The project for EC500 K1

#### Files Description (in the order of running)

*LUNA_mask_extraction.py* generates masks on lungs for segmenting lungs in the following step.

*LUNA_segment_lung.py* uses the masks from the last step to implement the segmentation

*LUNA_unet_seg.py* applies our ConvNet model to provide masks of nodule candidates of the segmented lungs

*csv_to_pickle.py* converts the .csv to .pkl which is used in *LUNA_classify_nodes.py*

*LUNA_classify_nodes.py* measures the features of the nodule candidates and feeds them into the classifier XGBoost
