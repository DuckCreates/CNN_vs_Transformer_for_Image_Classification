# Skin lesion Classification: CNN vs Vision Transformer
> **Status: Draft - work in progress**. This README currently only covers setup and the preprocessing stage. It will be updated as training and evaluation scripts are added.

### Overview
This project compares a ResNet18 CNN and a ViT-Tiny Vision Transformer for skin lesion classification using the HAM10000 dermoscopic image dataset.

### Requirements
* Python 3.10
* macOS with Apple Silicon for MPS -accelerated training

### Setup
1. Create and activate a cobda environment:
```bash
conda create -n skin-lesion python = 3.10
conda activate skin-lesion
```
2. Install dependencies:
```bash
conda install pytorch torchvision torchaudio -c pytorch
pip install timm scikit-learn matplotlip seaborn pillow pandas
```
3. Verify MPS is available
```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```
This should print True

### Dataset
This project uses the [HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) from Kaggle. Download it manually and place it in a folder named DataSet/archive/ in the project root
```
project-root/
├── 01_preprocess.py
├── DataSet/
│   └── archive/
│       ├── HAM10000_metadata.csv
│       ├── HAM10000_images_part_1/
│       └── HAM10000_images_part_2/
```

### Running the preprocessing script
```bash
python 01_preprocess.py
```

**What it does**
1. Loads HAM10000_metadata.csv(10,015 image records)
2. Matches each row to its actual image
3. Prints the class distribution (the dataset is known to be imbalanced - ~ 67% of images are the nv class)
4. Splits the dataset into train/validation/test sets (70/15/15), splitting by **lesion Id** rather than by image, to avoid the same lesion's photos leaking across splits
5. Computes class weights to help counteract the class imbalance during model training
6. Saves two output files:
   * ham10000_processed_metadata.cvs - full metadata with image paths and split assignments
   * class_weights.csv - per-class weights for use in the training loss function

### Expected output
```
Loaded 10015 image records
Unique lesions 7470
Images not found: 0
Remaining usable records: 10015
 
Images per split:
train   7002
val     1532
test    1481

Lesions per split:
train   5229
test    1121
val     1120

Preprocessing completed
```

### Project structure
```
project-root/
├── 01_preprocess.py                    # dataset loading, splitting, class weights
├── ham10000_processed_metadata.csv     # generated — not tracked in git
├── class_weights.csv                   # generated — not tracked in git
├── DataSet/                            # dataset — not tracked in git
├── .gitignore
└── README.md
```
