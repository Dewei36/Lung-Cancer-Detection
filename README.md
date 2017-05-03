# Lung-Cancer-Detection
The project for EC500 K1

#### Preprocessing
First run *step1_preprocess_ndsb.py*. This will extract all the ndsb dicom files , scale to 1x1x1 mm, and make a directory containing .png slice images. Lung segmentation mask images are also generated. They will be used later in the process for faster predicting.
Then run *step1_preprocess_luna16.py*. This will extract all the LUNA source files , scale to 1x1x1 mm, and make a directory containing .png slice images. Lung segmentation mask images are also generated. This step also generates various CSV files for positive and negative examples.

The nodule detectors are trained on positive and negative 3d cubes which must be generated from the LUNA16 and NDSB datasets. *step1b_preprocess_make_train_cubes.py* takes the different csv files and cuts out 3d cubes from the patient slices. The cubes are saved in different directories. *resources/step1_preprocess_mass_segmenter.py* is to generate the mass u-net trainset. It can be run but the generated resized images + labels is provided in this archive so this step does not need to be run. However, this file can be used to regenerate the traindata.

#### Training neural nets
First train the 3D convnets that detect nodules and predict malignancy. This can be done by running 
the *step2_train_nodule_detector.py* file. This will train various combinations of positive and negative labels. The resulting models (NAMES) are stored in the ./workdir directory and the final results are copied to the models folder.
The mass detector can be trained using *step2_train_mass_segmenter.py*. It trains 3 folds and final models are stored in the models (names) folder. Training the 3D convnets will be around 10 hours per piece. The 3 mass detector folds will take around 8 hours in total

#### Predicting neural nets
Once trained, the models are placed in the ./models/ directory.
From there the nodule detector *step3_predict_nodules.py*  can be run to detect nodules in a 3d grid per patient. The detected nodules and predicted malignancy are stored per patient in a separate directory. 
The masses detector is already run through the *step2_train_mass_segmenter.py* and will stored a csv with estimated masses per patient.
