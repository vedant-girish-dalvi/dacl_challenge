import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

IMAGE_HEIGHT = 704
IMAGE_WIDTH = 512
IMAGE_DIR = [r'./images/train', r'./images/validation', r'./images/testdev']
MASK_DIR = [r'./masks_grayscale/train',r'./masks_grayscale/validation']
CLASSES = ['Crack', 'ACrack', 'Wetspot', 'Efflorescence', 'Rust', 'Rockpocket', 'Hollowareas', 'Cavity', 'Spalling', 'Graffiti', 'Weathering', 'Restformwork', 'ExposedRebars', 'Bearing', 'EJoint', 'Drainage', 'PEquipment', 'JTape', 'WConccor']

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

class Segmentation_Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")    #grayscale
        
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
            # augmented = self.transform(image=image, mask=mask)
            # image = augmented["image"]
            # mask = augmented["mask"]

        # print(f"mask before:{mask.shape}")
        # mask = mask.squeeze(1)
        # print(f"mask after:{mask.shape}")
        # mask = torch.tensor(mask, dtype=torch.long)
        # mask = mask.clone().detach().requires_grad_(True)
        # mask = F.one_hot(mask, num_classes=19).squeeze(1)
        # mask = mask.squeeze(1)
        # print(mask.shape)
        # .permute(2, 0, 1).float()  # Shape: [19, 256, 256]
        
        return image, mask


#Augmentations for training and validation
# train_transforms = A.Compose(
#     [
#         A.Resize(width=IMAGE_WIDTH, height=IMAGE_HEIGHT),
#         A.RandomCrop(width=IMAGE_WIDTH, height=IMAGE_HEIGHT),
#         A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.1),
#         A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
#         A.OneOf([
#             A.Blur(blur_limit=3, p=0.5),
#             A.ColorJitter(p=0.5),
#         ], p=1.0),
#         ToTensorV2
#     ]
# )

# val_transforms = A.Compose(
#     [
#         A.Resize(width=IMAGE_WIDTH, height=IMAGE_HEIGHT),
#         A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
#         ToTensorV2
#     ]
# )

image_transforms = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor()
])

mask_transforms = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor()
])

train_dataset = Segmentation_Dataset(image_dir=IMAGE_DIR[0], mask_dir=MASK_DIR[0], transform=image_transforms)
validation_dataset = Segmentation_Dataset(IMAGE_DIR[1], MASK_DIR[1], transform=mask_transforms)
# test_dataset = Segmentation_Dataset(IMAGE_DIR[2], MASK_DIR[2], transform=None, mask_transform=None)

# image, mask = validation_dataset.__getitem__(750)
# fig, axes = plt.subplots(1, 2, figsize=(8, 4))
# axes[0].imshow(image.permute(1, 2, 0))  # Convert (C, H, W) to (H, W, C)
# axes[0].set_title("Image")
# axes[0].axis("off")
# axes[1].imshow(mask.squeeze(), cmap="gray")  # Remove channel dimension if present
# axes[1].set_title("Mask")
# axes[1].axis("off")
# plt.show()



# if __name__ == "__main__":
    
#     train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
#     validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
#     for images, masks in train_dataloader:
#         print("Image batch shape:", images.shape)
#         print("Mask batch shape:", masks.shape)
#         break