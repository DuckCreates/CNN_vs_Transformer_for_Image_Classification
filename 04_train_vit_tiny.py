#Imports
import pandas as pd
import torch                                            # Core pytorch library
import torch.nn as nn                                   # Neural Network layers
import torch.optim as optim        # Base classses for budiling a data pipeline
import timm

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
#Same Seed
torch.manual_seed(42)

# Device setup: Use MPS if avalilable, else COP
device = torch.device("cpu")
print(f"Using devive: {device}")

# Load processed metadata
df = pd.read_csv("ham10000_processed_metadata.csv")

class_names = sorted(df["dx"].unique())
class_to_idx = {name: idx for idx, name in enumerate(class_names)}
num_classes = len(class_names)
print(f"Classes: {class_to_idx}")

# Dataset class (same as 02_dataset.py)
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

# Datasets and dataloaders
batch_size = 32
#train_dataset = HAM10000Dataset(df[df["split"] == "train"], transform = train_transform)

# Subest the training data for dataset-szie ablation ---
# Change this value to 0.25, 0.5, 0.75 or 1.0 for each experiment run
train_faction = 0.75
if train_faction < 1.0:
# saample a fractio of the training data, stratified by class so the class
# distribution stays representaive enen in smaller subsets
    train_df_subset = df[df["split"] == "train"].groupby("dx", group_keys = False).apply(lambda x: x.sample(frac=train_faction, random_state = 42))
    print(f"Using {train_faction*100:.0f}% of training data: {len(train_df_subset)} images")
else:
    train_df_subset = df[df["split"] == "train"]
    print(f"Using 100% of training data: {len(train_df_subset)} images")
train_dataset = HAM10000Dataset(train_df_subset, transform=train_transform)

val_dataset = HAM10000Dataset(df[df["split"] == "val"], transform = eval_transform)
train_loader = DataLoader(train_dataset,batch_size = batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# load the pre-trained ViT-Tiny
# timm's model naming: "vit_tiny_patch16_224" = ViT-Tiny, 16 x 16 patches, 224 x 224 input
# pretrained = TRUE lpades ImageNet-pretrained weights, same transfer learning
model = timm.create_model("vit_tiny_patch16_224", pretrained = True, num_classes = num_classes)

# timms model lets us pass num_classes directly and it handles replacing the final classification head for us
model = model.to(device)

print(model.head) #check

# Loss function with class weights
# load the classes weights we computed during preprocessing, so mistakes on rare classes are penalized more heavily tha mistakes on the dominant class (nv)
weights_df = pd.read_csv("class_weights.csv", index_col = 0)

# Reorder the weights to match class_to_idx order (0=akiec, 1=bcc, ..., 6= vasc)
# since the CSV might not be in that order
class_weights_ordered = [weights_df.loc[name, "weight"] for name in class_names]
class_weights_tensor = torch.tensor(class_weights_ordered, dtype = torch.float32).to(device)
print(f"Class weights tensor: {class_weights_tensor}")

# Cross entropyloss is standard for multi-class classification weight == applies our class weights so rares classes count more.
criterion = nn.CrossEntropyLoss(weight = class_weights_tensor)

# Optimizer
# AdamW is a standard, reliable choice for fine-tuning pretrained models.
optimizer = optim.AdamW(model.parameters(), lr = 3e-5)
print("Loss function and optimizer set up.")

# Training loop
num_epochs = 20 # start small to test everything works, increase later
best_val_acc = 0.0 # Will update with the best val - track the model

# Early stoppin
patience = 3
epoch_no_improve = 0

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()       # clear gradients from the previous step
        outputs = model(images)     # forward pass: get predictions
        loss = criterion(outputs, labels) # Compare predictions to true labels
        loss.backward()             # backward pass: compute gradients
        optimizer.step()            # update model weights
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1) # get the predictec class (highest score)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    train_loss = running_loss / total
    train_acc = correct / total
        
        # validation phase
    model.eval() # tells the model it's in evaluation mode
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():   # don't compute gradients during validation (faster, saves memory)
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    val_loss = val_loss / val_total
    val_acc = val_correct / val_total
    # save the model if this is the best accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_vit_tiny_frac75.pth")
        print(f"-> New best Model saved (val_acc: {val_acc:.4f})")
        epoch_no_improve = 0
    else:
        epoch_no_improve += 1
    
    print(f"Epoch {epoch+1}/{num_epochs} | " f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | " f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    if epoch_no_improve >= patience:
        print(f"\nEarly Stopping trigger: epoch {epoch+1} (no improvement {patience} to epoch).")
        break
