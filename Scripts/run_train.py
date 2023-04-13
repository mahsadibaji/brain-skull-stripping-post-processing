import torch
import argparse
from model import *
from data_loader import *
from training import train_unet
from verbose_utils import plot_samples_source_loaders
from torch.optim.lr_scheduler import StepLR

# python main.py --batch_size 1 --source_dev_images ./Data-split/source_train_set_neg.csv --source_dev_masks ./Data-split/source_train_set_masks_neg.csv  --target_dev_images ./Data-split/target_train_set.csv --verbose True

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size,  number of images in each iteration during training')
    parser.add_argument('--epochs', type=int, default=100, help='total epochs')
    parser.add_argument('--val_split', type=float, default=0.3, help='Vaal split')
    parser.add_argument('--results_dir', type=str, default ="./results/", help='results directory')
    parser.add_argument('--source_dev_images', type=str, help='path to source dev images')
    parser.add_argument('--source_dev_masks', type=str, help='path to source dev masks')
    parser.add_argument('--verbose', type=bool, default=False, help='verbose debugging flag')
    
    args = parser.parse_args()

    root_dir = args.results_dir # Path to store results
    verbose = args.verbose # Debugging flag
    
    # Set our data loaders - supervised training with no domain adaptation
    source_ds_train, source_train_loader, \
    source_ds_val, source_val_loader = load_data(args.source_dev_images,\
                                                     args.source_dev_masks, root_dir,\
                                                     batch_size = args.batch_size, val_split = args.val_split, verbose = verbose)
    
    # Inspecting output of source domain data loaderss
    if verbose:
        plot_samples_source_loaders(source_train_loader, source_val_loader, root_dir)

    model = Unet()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    model = model.cuda()
        
    # to do - learning rate scheduler, early stoppin, weights and biases

    # Learning rate decay scheduler
    scheduler = StepLR(optimizer, step_size=40, gamma=0.5)

    print("Start of training...")
    train_unet(source_train_loader, source_val_loader,\
                model, optimizer, scheduler, args.epochs, root_dir)
    
    print("End of training.")
        