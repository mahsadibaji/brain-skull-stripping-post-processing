from monai.losses import DiceLoss
import torch
import matplotlib.pylab as plt
import numpy as np
import torch.nn as nn
import numpy as np

def train_unet(train_loader, val_loader, model, optimizer, scheduler, max_epochs, root_dir):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.train() # training mode - layer like dropout are active
    
    best_val_loss = 1.0

    loss_object = DiceLoss(to_onehot_y = True)
    for epoch in range(1,max_epochs +1):
        train_loss = 0.0
        val_loss = 0.0
    
        print("Epoch ", epoch)
        print("Train:", end ="")
        for step, batch in enumerate(train_loader):
            img, brain_mask = (batch["img"].cuda(), batch["brain_mask"].cuda())

            optimizer.zero_grad()

            pred_tissue_mask = model(img) # forward pass

            loss = loss_object(pred_tissue_mask,brain_mask)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            print("=", end = "")

            

        train_loss = train_loss/(step+1)
        
        print()
        print("Val:", end ="")
        model.eval() # innference mode - layers like dropout get disabled
        with torch.no_grad():
                for step, batch in enumerate(val_loader):
                    brain_img, brain_mask = (batch["img"].cuda(), batch["brain_mask"].cuda())
                    
                    pred_tissue_mask = model(brain_img)

                    loss = loss_object(pred_tissue_mask,brain_mask)
                    val_loss += loss.item()
                    print("=", end = "")
                print()
                val_loss = val_loss/(step+1)
        
        img = brain_img.cpu()
        pred_tissue_mask = pred_tissue_mask.cpu()
        plt.figure()
        plt.subplot(121)
        plt.imshow(img.numpy()[0,0,32,:,:], cmap = "gray")
        plt.subplot(122)
        plt.imshow(img.numpy()[0,0,32,:,:], cmap = "gray")
        plt.imshow(np.argmax(pred_tissue_mask.numpy(),axis = 1)[0,32,:,:], alpha = 0.4)
        plt.savefig(root_dir +"val_sample_epoch_" + str(epoch) + ".png")

        print("Training epoch ", epoch, ", train loss:", train_loss, ", val loss:", val_loss)

        if val_loss < best_val_loss:
            print("Saving model")
            torch.save(model.state_dict(), root_dir + "unet.pth")    
            best_val_loss = val_loss
    return
