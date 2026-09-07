#Imports
import pandas as pd
import torch
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# The comparsion
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Device setup: Use MPS if avalilable, else COP
device = torch.device("cpu")
print(f"Using devive: {device}")

# Load preocessd metadata
df = pd.read_csv("ham10000_processed_metadata.csv")
class_names = sorted(df["dx"].unique())
class_to_idx = {name: idx for idx, name in enumerate(class_names)}
num_classes = len(class_names)
print(f"Classes: {class_to_idx}")

# Dataset class
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

# Transform (same as 02_dataset.py)
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
# Training transform includes augmentation to help the model generalize and reduce overfitting on the training set
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
                                      ])
# test dataset and loader
test_dataset = HAM10000Dataset(df[df["split"] == "test"], transform = eval_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Test set size: {len(test_dataset)}, Test batches:{len(test_loader)}")

# Load resnet final, 100% data version)
resnet_model = models.resnet18(weights = None)
num_features = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_features, num_classes)
resnet_model.load_state_dict(torch.load("best_resnet18_Ir1e-4.pth", map_location= device))
resnet_model = resnet_model.to(device)
resnet_model.eval() # set to evaluation mode
print("resNet18 loaded.")

# Load ViT tiny
vit_model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes)
vit_model.load_state_dict(torch.load("best_vit_tiny_lr3e-4.pth", map_location=device))
vit_model = vit_model.to(device)
print("ViT-Tiny loaded.")

# function to run a mdoel on the test set and collect predictions
def get_predictions(model, loader):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    return all_labels, all_preds

# get predictions from both models
print("\nEvaluating ResNet18 on test set...")
resnet_labels, resnet_preds = get_predictions(resnet_model, test_loader)

print("\nEvaluating ViT-Tiny on test set...")
vit_labels, vit_preds = get_predictions(vit_model, test_loader)

print("\nDone.")

# Classification reports (precision, recall, f1 per class)
print("\n" + "="*60)
print("RESNET18 - Test Set Classification Report")
print("\n" + "="*60)
print(classification_report(resnet_labels, resnet_preds, target_names = class_names, digits=4))

print("\n" + "="*60)
print("VIT-TINY - Test Set Classification Report")
print("\n" + "="*60)
print(classification_report(vit_labels, vit_preds, target_names = class_names, digits=4))

# Confussion matrcis
resnet_cm = confusion_matrix(resnet_labels, resnet_preds)
vit_cm = confusion_matrix(vit_labels, vit_preds)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(resnet_cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels = class_names, ax=axes[0])
axes[0].set_title("ResNet18 - Confusion Matrix (test Set)")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(vit_cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels = class_names, ax=axes[1])
axes[1].set_title("ViT-tiny - Confusion Matrix (test Set)")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.show()
print("\nConfusion Matrices save to cinfusion_matrices.png")

