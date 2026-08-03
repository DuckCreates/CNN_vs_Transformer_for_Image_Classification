#Imports
import pandas as pd                                     # For reading the CSV
import torch                                            # Core pytorch library
from torch.utils.data import Dataset, DataLoader        # Base classses for budiling a data pipeline
from torchvision import transforms                       # Image resizing/normalization / augmentation
from PIL import Image                                   # for opening image files

# Load the CSV we createing in the preprocessing file
df = pd.read_csv("ham10000_processed_metadata.csv")
print(f"Loaded {len(df)} records")
print(df["split"].value_counts())

# Maps each diagnosis label to a number, since neural networks need numerical labels, not text
# sorted() ensures this mapping is always the same order every time you run it
class_names = sorted(df["dx"].unique())
class_to_idx = {name: idx for idx, name in enumerate(class_names)}
print(f"\nClass mapping: {class_to_idx}")

# A Dataset tells PyTorch how to get ONE item at a time.
# PyTorch's DataLoader will use this to build batches automatically.
class HAM10000Dataset(Dataset):
    def __init__(self, dataframe, transform = None):
        # Store the relevant rows and the transform piple to apply to each image.
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
    
    def __len__(self):
        # Pytorch needs to know how many items are in the dataset
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # This called automatically by PyTotch whenever it wants item number `idx`. it must return (image_tensor, label).
        row = self.dataframe.iloc[idx]
        # open the image file and make sure it's RGB
        image = Image.open(row["image_path"]).convert("RGB")
        # APply resizing/augmentation/normalization
        if self.transform:
            image = self.transform(image)
        # convert the text label into its numeric index
        label = class_to_idx[row["dx"]]
        return image, label

# --- Transformer ---
# ImageNet mean/std, since btoh ResNet18 and ViT-Tiny were pretrained on ImageNet and expect inputs normalized this way.
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
# Training transform includes augmentation to help the model generalize and reduce overfitting on the training set
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),                  # converts image to a Pythorch tensor
    transforms.Normalize(imagenet_mean, imagenet_std),
                                      ])

# Validation / test transform: NO augmentation. We want to evaluate on, consistent, unmodified images so results are comparable and reproducible.
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
                                     ])
# -- Create the three dataset splits --
train_dataset = HAM10000Dataset(df[df["split"] == "train"], transform= train_transform)
val_dataset = HAM10000Dataset(df[df["split"] == "val"], transform = eval_transform)
test_dataset = HAM10000Dataset(df[df["split"] == "test"], transform = eval_transform)

print(f"\nTrain dataset size: {len(train_dataset)}")
print(f"Val dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# -- Test laoding a single image to make sure everything works --
image, label = train_dataset[0]
print(f"\nSample image shape: {image.shape}") # should be [3, 224, 224]
print(f"Sample label: {label}")

# Dataloader wraps a Dataset and Handles batch + shuffling automatically.
# Batch_size: how many images are processed together in one training step
# Shuffle = True for training so the model doesn't see the same order every epoch which could cause it to learn spurious patterns from the order
# Shuffle = False for val/test 9order doesn't matter, and keeping it fixed makes debugging /comparsion easier)
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

images, labels = next(iter(train_loader))
print(f"\nBatach of images shape: {images.shape}")  # Should be [32, 3., 224, 224]
print(f"Batch of labels shape: {labels.shape}")     # Should be [32]
