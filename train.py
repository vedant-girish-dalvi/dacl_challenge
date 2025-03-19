import os
import torch
import numpy as np
import albumentations as A
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import SegmentationModel
from dataset import train_dataset, validation_dataset
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, iou_score

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyperparameters
NUM_CLASSES = 19
EPOCHS = 10        
LEARNING_RATE = 0.001
IMAGE_SIZE = 800   
BATCH_SIZE = 32
NUM_WORKERS=4
PIN_MEMORY=True
LOAD_MODEL = False
ARCHITECTURE = 'Unet'
ENCODER = 'resnet34'
WEIGHTS = 'imagenet'
INPUT_CHANNELS = 3

writer = SummaryWriter("runs/segmentation_experiment")

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

def train_model(model, train_loader, val_loader, loss_function, optimizer, num_epochs, device="cuda:0", checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.to(device)
    best_val_loss = float('inf')

    print(f"---------------Model training on {device}-------------------")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, masks in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"): 
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            print(f"outputs:{outputs.shape}, masks: {masks.shape}")
            loss = loss_function(outputs, masks)
            loss.backward() 
            optimizer.step()
            train_loss += loss.item()
            tqdm.set_postfix(loss=loss.item()) 

            writer.add_scalar('Loss/Train_Batch', loss.item())
        
        avg_train_loss = train_loss/len(train_loader)
        writer.add_scalar('Loss/Train',avg_train_loss,epoch)

        model.eval()
        val_loss = 0
        iou_scores = []
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = loss_function(outputs, masks)
                # print(outputs.shape, masks.shape)
                val_loss += loss.item()

                iou = iou_score(outputs, masks, num_classes=19)
                iou_scores.append(iou)

        avg_val_loss = val_loss / len(val_loader)
        avg_iou = np.mean(iou_scores)

        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('IoU/Validation', avg_iou, epoch)
    
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, IoU: {avg_iou: .4f}")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'best_model.pth'))
            print(f" Best model saved at epoch {epoch+1} with val loss {avg_val_loss:.4f}")

    print("Training Complete!")
    writer.close()

        

def main():

    model = SegmentationModel(arch=ARCHITECTURE, encoder=ENCODER, weights=WEIGHTS, num_classes=NUM_CLASSES).to(device=DEVICE)
    loss = smp.losses.DiceLoss(mode='multilabel')
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(model, train_dataloader, validation_dataloader, loss, optimizer, num_epochs=EPOCHS, checkpoint_dir="./checkpoints")

if __name__ == '__main__':
    main()