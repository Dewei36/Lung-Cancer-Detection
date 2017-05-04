import numpy as np
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
from glob import glob

working_path = "/Users/weichaozhou/Documents/Boston University/EC500/DSB3Tutorial/LUNA2016/"
sample_list=glob(working_path+"images_*.npy")

for IMAGE in sample_list:
    img_current = np.load(IMAGE).astype(np.float64) 
    print ("on image", IMAGE)
    for i in range(len(img_current)):
        image = img_current[i]
        mean = np.mean(image)
        std = np.std(image)
        image = image-mean
        image = image/std
        middle = image[100:400,100:400] 
        mean = np.mean(middle)  
        max = np.max(image)
        min = np.min(image)
        image[image==max]=mean
        image[image==min]=mean
       
        kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
        centers = sorted(kmeans.cluster_centers_.flatten())
        thr = np.mean(centers)
        processed_img = np.where(image<thr,1.0,0.0)  # threshold the image
        #Morphology processing
        erosion_result = morphology.erosion(processed_img,np.ones([4,4]))
        open_result = morphology.dilation(erosion_result,np.ones([10,10]))
        labels = measure.label(open_result)
        label_vals = np.unique(labels)
        regions = measure.regionprops(labels)
        perfect_labels = []
        for prop in regions:
            B = prop.bbox
            if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
                perfect_labels.append(prop.label)
        mask = np.ndarray([512,512],dtype=np.int8)
        mask[:] = 0
        
        for N in perfect_labels:
            mask = mask + np.where(labels==N,1,0)
        mask = morphology.dilation(mask,np.ones([10,10])) 
        img_current[i] = mask
    np.save(IMAGE.replace("images","lungmask"),img_current)
    




sample_list=glob(working_path+"lungmask_*.npy")
out_images = []      
out_nodule_masks = []   
for filename in sample_list:
    print ("working on file ", filename)
    img_current = np.load(filename.replace("lungmask","images"))
    masks = np.load(filename)
    nodule_masks = np.load(filename.replace("lungmask","masks"))
    for i in range(len(img_current)):
        mask = masks[i]
        nodule_mask = nodule_masks[i]
        image = img_current[i]
        new_size = [512,512]   
        image= mask*image        
        
        new_mean = np.mean(image[mask>0])  
        new_std = np.std(image[mask>0])
        
        min_intensity = np.min(image)     
        image[image==min_intensity] = new_mean-1.2*new_std  
        image = image-new_mean
        image = image/new_std
        labels = measure.label(mask)
        regions = measure.regionprops(labels)
       
        min_row = 512
        max_row = 0
        min_col = 512
        max_col = 0
        for prop in regions:
            B = prop.bbox
            if min_row > B[0]:
                min_row = B[0]
            if min_col > B[1]:
                min_col = B[1]
            if max_row < B[2]:
                max_row = B[2]
            if max_col < B[3]:
                max_col = B[3]
        width = max_col-min_col
        height = max_row - min_row
        if width > height:
            max_row=min_row+width
        else:
            max_col = min_col+height
       
        image = image[min_row:max_row,min_col:max_col]
        mask =  mask[min_row:max_row,min_col:max_col]
        if max_row-min_row <5 or max_col-min_col<5:  
            pass
        else:
          
            mean = np.mean(image)
            image = image - mean
            min = np.min(image)
            max = np.max(image)
            image = image/(max-min)
            new_img = resize(image,[512,512])
            new_nodule_mask = resize(nodule_mask[min_row:max_row,min_col:max_col],[512,512])
            out_images.append(new_img)
            out_nodule_masks.append(new_nodule_mask)

no_images = len(out_images)

final_images = np.ndarray([no_images,1,512,512],dtype=np.float32)
final_masks = np.ndarray([no_images,1,512,512],dtype=np.float32)
for i in range(no_images):
    final_images[i,0] = out_images[i]
    final_masks[i,0] = out_nodule_masks[i]

rand_i = np.random.choice(range(no_images),size=no_images,replace=False)
test_i = int(0.2*no_images)
np.save(working_path+"trainImages.npy",final_images[rand_i[test_i:]])
np.save(working_path+"trainMasks.npy",final_masks[rand_i[test_i:]])
np.save(working_path+"testImages.npy",final_images[rand_i[:test_i]])
np.save(working_path+"testMasks.npy",final_masks[rand_i[:test_i]])


