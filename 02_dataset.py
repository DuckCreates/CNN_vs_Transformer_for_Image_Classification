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
