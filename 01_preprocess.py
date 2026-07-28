#Imports
import os                                               # Checks for file paths
import pandas as pd                                     # For loading the metadata table
from sklearn.model_selection import train_test_split    # Used to create the Test/Val/Train sets

# Path to the relative dataset (NOTE Data/archive must be in the same folder as scripts)
dataset_path = "DataSet/archive"
# Safety Check: Stop script with error message if path can't be found
assert os.path.exists(dataset_path), f"Could not find {dataset_path}"
print(f"Using dataset at {dataset_path}")

# Builds the path to the metadata.csv (does it safely using os.path.join())
metadata_path = os.path.join(dataset_path, "HAM10000_metadata.csv")
# Load the metadat in Pandas dataframe
df = pd.read_csv(metadata_path)

# lend(df) is the number of rows / images in the dataset
# df['lesion_id].nunique() pulls that column with only unique ids.
print(f"Loaded {len(df)} image records")
print(f"Unique lesions {df['lesion_id'].nunique()}")

# HAM10000's images are split across two folders.
# Both and build a list of all folders that actually contain images
image_dirs = []
for root, dirs, files in os.walk(dataset_path):
    for d in dirs:
        if "images" in d.lower():
            image_dirs.append(os.path.join(root, d))
print(f"Found image folders: {image_dirs}")

# For a given image_id, search through each image folder and return the full path to the matching .jpg file, or None if it's not found in any of them
def find_image_path(image_id):
    for d in image_dirs:
        candidate = os.path.join(d, f"{image_id}.jpg")
        if os.path.exists(candidate):
            return candidate
    return None

# .apply() runs find_image_path() on every row's image_id and store the result in a new column classed"image_path"
df["image_path"] = df["image_id"].apply(find_image_path)
# Check how many rows failed to find a matching image file
missing = df["image_path"].isna().sum()
print(f"Image not found: {missing}")

# Drop any rows where we couldn't find the image, since we can't use them for training
df = df.dropna(subset = ["image_path"]).reset_index(drop = True)
print(f"Remaining usable records: {len(df)}")

# See how many images belong to each diagnosis categorys (Shoulds class imbalancement)
print("\nClass Counts:")
print(df["dx"].value_counts())
print("\nClass Percentages:")
print((df["dx"].value_counts(normalize= True)*100).round(2))

# Get one row per unique lesion, since we want to plit by lesion to avoid the same
# lesion appearing in both sets
lesion_df = df.drop_duplicates(subset = "lesion_id") [["lesion_id", "dx"]]

# First split: 70% train 30% temp
# stratify = ensures each split keeps the same class proprtions as the original dataset
# Random-stae fixes the randomness so results are reproducible
train_lesion, temp_lesions = train_test_split(lesion_df, test_size = 0.30, stratify = lesion_df["dx"], random_state = 42,)

#second split: divide the remaining 30% in half -> 15% val, 15% test
val_lesions, test_lesions = train_test_split(temp_lesions, test_size = 0.50, stratify = temp_lesions["dx"], random_state = 42,)

# convert to sets for fast lookup
train_ids = set(train_lesion["lesion_id"])
val_ids = set(val_lesions["lesion_id"])
test_ids = set(test_lesions["lesion_id"])

# given a lesion_id, return which split it belongs to
def assign_split(lesion_id):
    if lesion_id in train_ids:
        return "train"
    elif lesion_id in val_ids:
        return "val"
    elif lesion_id in test_ids:
        return "test"
    return

# Apply this to every row in the full image-level dataframe, so each image nows knows which split it belongs to
df["split"] = df["lesion_id"].apply(assign_split)
print("\nImages per split:")
print(df["split"].value_counts())
print("\nLesions per split:")
print(df.drop_duplicates(subset = "lesion_id")["split"].value_counts())

# Compute class weights to help the model during training
# Classes with fewer images get a HIGHer weight, so mistakes on rare classes "count more" in the loss function. this helps counteract the imbalacne we saw earlie
# Only look at the training set for this
train_df = df[df["split"] == "train"]
class_counts = train_df["dx"].value_counts().sort_index()

total = class_counts.sum()
num_classes = len(class_counts)

# Inverse frequency weighting: rarer classes get bugger weights
# Formula: weight = total_samples / (num_classes * count_for_this_class)
class_weights = total / (num_classes * class_counts)

# Normalize so the weights averaget to 1
class_weights = class_weights / class_weights.mean()
print("\nClass weights:")
print(class_weights)

# Save the weights to a CSV so the training script can load them laster
class_weights.to_csv("class_weights.csv", header=["weight"])

#same the complete datadraem (with image paths and split assignments) to a CSV. The training script will load THIS file instead of redoing all the steps every time
output_csv = "ham10000_processed_metadata.csv"
df.to_csv(output_csv, index = False)
print(f"\nSaved processed metadate to: {output_csv}")
print("Preprocessing complete.")
