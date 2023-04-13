import monai
from monai.data import Dataset, DataLoader
from monai.transforms import (LoadImaged, EnsureChannelFirstd, ScaleIntensityd, \
                              RandAxisFlipd, RandGaussianNoised, RandGibbsNoised, \
                              RandSpatialCropd, Compose, RandAdjustContrastd, RandRotated)
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from monai.data.utils import pad_list_data_collate

source_transforms = Compose(
    [
        LoadImaged(keys=["img", "brain_mask"]),
        EnsureChannelFirstd(keys=["img",  "brain_mask"]),
        ScaleIntensityd(keys=["img"], minv=0.0, maxv=1.0),
        RandSpatialCropd(keys=["img","brain_mask"], roi_size=(112, 112, 112), random_size=False),
        RandAxisFlipd(keys=["img", "brain_mask"], prob = 0.5),
        RandRotated(keys=["img", "brain_mask"], range_x=np.pi / 4, prob=0.5, keep_size=True, mode="nearest"),
        RandGaussianNoised(keys = ["img"], prob=0.2, mean=0.0, std=0.05),
        RandGibbsNoised(keys=["img"], prob = 0.2, alpha = (0.1,0.6)),
        RandAdjustContrastd(keys=["img"], prob=0.5)
    ]
)


def load_data(source_dev_images_csv, source_dev_masks_csv, root_dir,
              batch_size = 1, val_split = 0.2, verbose = False):


    source_dev_images = pd.read_csv(source_dev_images_csv)
    source_dev_masks = pd.read_csv(source_dev_masks_csv)
    if verbose:
        print("source images size:",source_dev_images.size)
        print("source masks size:", source_dev_masks.size)
    assert source_dev_images.size == source_dev_masks.size

    if verbose:
        print("Shape source images:", source_dev_images.shape)
        print("Shape source masks:",  source_dev_masks.shape)
    
    indexes_source = np.arange(source_dev_images.shape[0])
    
    np.random.seed(100)  
    np.random.shuffle(indexes_source)
    source_dev_images = np.array(source_dev_images["filename"])[indexes_source]
    source_dev_masks = np.array(source_dev_masks["filename"])[indexes_source]
    
    ntrain_samples = int((1 - val_split)*indexes_source.size)
    source_train_images = source_dev_images[:ntrain_samples]
    source_train_masks = source_dev_masks[:ntrain_samples]

    source_val_images = source_dev_images[ntrain_samples:]
    source_val_masks = source_dev_masks[ntrain_samples:]

    if verbose:
        print("Source train set size:", source_train_images.size)
        print("Source val set size:", source_val_images.size)

        np.savetxt(root_dir+'val_set.csv',source_val_images,fmt='%s')
        np.savetxt(root_dir+'val_set_mask.csv',source_val_masks,fmt='%s')
        
    # Putting the filenames in the MONAI expected format - source train set
    filenames_train_source = [{"img": x, "brain_mask": y}\
                              for (x,y) in zip(source_train_images, source_train_masks)]
       
    source_ds_train = monai.data.Dataset(filenames_train_source,
                                         source_transforms)

    source_train_loader = DataLoader(source_ds_train, 
                                    batch_size=batch_size, 
                                    shuffle = True, 
                                    num_workers=0, 
                                    pin_memory=True, 
                                    collate_fn=pad_list_data_collate)

    # Putting the filenames in the MONAI expected format - source val set
    filenames_val_source = [{"img": x, "brain_mask": y}\
                              for (x,y) in zip(source_val_images, source_val_masks)]
       
    source_ds_val = monai.data.Dataset(filenames_val_source,
                                         source_transforms)
                                         
    source_val_loader = DataLoader(source_ds_val, 
                                    batch_size=batch_size, 
                                    shuffle = True, 
                                    num_workers=0, 
                                    pin_memory=True, 
                                    collate_fn=pad_list_data_collate)


    return source_ds_train, source_train_loader, source_ds_val, source_val_loader