import torch
import matplotlib.pyplot as plot
import pandas as pd
import numpy as np
import os

import monai
from monai.utils import set_determinism
from monai.data import Dataset, DataLoader, NibabelReader
from monai.losses import DiceLoss
from monai.metrics import DiceMetric 
from monai.inferers import sliding_window_inference
from monai.data.utils import pad_list_data_collate
from monai.transforms import (LoadImaged, EnsureChannelFirstd, ScaleIntensityd, \
                            Compose, DivisiblePadd, NormalizeIntensityd, Orientationd,AsDiscreted, ThresholdIntensityd,GaussianSmoothd)
import nibabel as nib
import argparse
from model import *
from verbose_utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default ="./test_results/", help='results directory')
    parser.add_argument('--model_path', type=str, default ="./train_results/unet.pth", help='path to saved model')
    parser.add_argument('--source_test_images', type=str, help='path to source test images')
    parser.add_argument('--source_test_masks', type=str, help='path to source test masks')
    parser.add_argument('--verbose', type=bool, default=False, help='verbose debugging flag')

    args = parser.parse_args()
    root_dir = args.results_dir
    verbose = args.verbose
    saved_model_path = args.model_path
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


    test_transform = Compose([
        LoadImaged(keys=["img","brain_mask"], reader=NibabelReader),
        EnsureChannelFirstd(keys=["img","brain_mask"]),
        ThresholdIntensityd(keys=["brain_mask"], threshold=0.5, above=False),  # Add this line
        DivisiblePadd(["img","brain_mask"], k=16),
        ScaleIntensityd(keys=["img"], minv=0.0, maxv=1.0),
    ])

    source_test_images = pd.read_csv(args.source_test_images)
    source_test_masks = pd.read_csv(args.source_test_masks)

    if verbose:
        print("source images size: ", source_test_images.size)
        print("source images size: ", source_test_masks.size)

    source_test_images = np.array(source_test_images["filename"])
    source_test_masks = np.array(source_test_masks["filename"])

    filename_test_source = [{"img":x, "brain_mask":y} for (x,y) in zip(source_test_images,source_test_masks)]

    source_ds_test = monai.data.Dataset(filename_test_source, test_transform)

    source_test_loader = monai.data.DataLoader(source_ds_test,
                                    batch_size=1, 
                                    shuffle=False, 
                                    num_workers=0, 
                                    pin_memory=True)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = Unet()
    model.cuda()
    model.load_state_dict(torch.load(saved_model_path))

    

    model.eval()
    loss_object = DiceLoss(to_onehot_y = True)

    test_score = 0
    test_loss = 0
    print("Start of testing...")
    with torch.no_grad():

        for i, batch in enumerate(source_test_loader):
            
            image, mask = (batch["img"].cuda(),batch["brain_mask"].cuda())
            print("mask shape:", mask.shape)
            print("image shape:", image.shape)
            file_path = source_test_images[i]
            filename = os.path.basename(file_path)

            roi_size = (96,96,96)
            sw_batch_size = 4
            pred = sliding_window_inference(image, roi_size, sw_batch_size, model)
            # pred = model(image)
            print("pred shape:", pred.shape)

            loss = loss_object(pred, mask)
            print(loss.item())
            test_loss += loss.item()
            

            pred = pred.argmax(1).squeeze().detach().cpu().numpy() #squeeze() to remove the channel dimension 
                                                                   #detach() to detach the tensor from the computation graph
            org_image = nib.load(file_path)
            org_affine = org_image.affine
            print(org_affine)
            org_shape = org_image.shape[:3]

            print(f" original image shape = {org_shape}")

            cropped_pred = symmetric_crop(pred, org_shape)

            print(f"Cropped prediction shape = {cropped_pred.shape}")

            output = nib.Nifti1Image(cropped_pred.astype(np.float32), org_affine)
            output_filename = filename.replace(".mri.nii.gz", ".unet.mask.nii.gz")
            print(output.affine)
            if verbose:
                assert output.affine.all() == org_affine.all()

            nib.save(output, root_dir + output_filename)

            plot_centre_slices(image.cpu().numpy()[0,0,:,:,:],cropped_pred, output_filename,  root_dir+'center_slices/')
            print(f'predicted batch = {i}')
            
        test_loss = test_loss/(i+1)
        
    print(f"Total Dice Loss: {test_loss}")
    print("End of testing")



