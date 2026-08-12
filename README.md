# Skin lesion Classification: CNN vs Vision Transformer
> **Status: Draft - work in progress**. This README currently only covers setup and the preprocessing stage. It will be updated as training and evaluation scripts are added.

### Overview
This project compares a ResNet18 CNN and a ViT-Tiny Vision Transformer for skin lesion classification using the HAM10000 dermoscopic image dataset.

### Requirements
* Python 3.10
* macOS with Apple Silicon(M4) for MPS -accelerated training

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
This project uses the [HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) from Kaggle. The **preprocessing script** will downlaod it automatically if it isn't already present at DataSet/archive/ in the project root. The first time this happens, kagglehub will open a brower tab asking you to log in to kaggle and authorize access - approve it there and the download will continue.
If you'd rather download it manually, place it in DataSet/archive/ so the structure looks like:
```
project-root/
├── 01_preprocess.py
├── DataSet/
│   └── archive/
│       ├── HAM10000_metadata.csv
│       ├── HAM10000_images_part_1/
│       └── HAM10000_images_part_2/
```
Requires the kagglehub package:
```bash
pip install kagglehub
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

**Expected outcome**
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
### Runing the Data piple script
```bash
python 02_dataset.py
```
**What it does**
1. Laods ham10000_processed_metadat.cvs
builds a class-to-index mapping for the 7 diagnosis categories:
```
{'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
```
3. Defines a custom Pytorch dataset that loads one image and its label at a time from CSV
defines separate image transforms:
    * Training: Resize to 224 x 224, random horizontal/vertical flip, random rotation, normalize using ImageNet mean /std
    * Validation/test: resize to 224x224 and normalize only
5. Create train_dataset, val_dataset, test_dataset, and their corresponding DataLoaders (batch size 32)
6. Runs a quick sanity check, loading one sample image and one batch to confirm the pipeline works end to end

**Expected outcome**
```
Loaded 10015 records
split
train    7002
val      1532
test     1481
Name: count, dtype: int64

Class mapping: {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}

Train dataset size: 7002
Val dataset size: 1532
Test dataset size: 1481

Sample image shape: torch.Size([3, 224, 224])
Sample label: 2

Batach of images shape: torch.Size([32, 3, 224, 224])
Batch of labels shape: torch.Size([32])
```
> Note you will see a warning, its okay it can be ignored

### Running the train resnet18

```bash
python 03_train_resnet18.py
```

**What it does**
1. Loads a pretrained ResNet18 (ImageNet weights) and replaces the final layer to output 7 classes instead of 1000
2. Sets up a class-weighted CorssEntropyLoss (using class_weights.csv from preprocessing) and AdamW optimizer
3. Trains with model checkpoint - saves the model to a .pth file whenever validation accuracy improves
4. Trains with early stopping - stop automatically if validation accuracy doesn improve for 3 consecutive epoches

**Results**

A Fixed random seed (torch.manual_ssed(42)) is used so runs are reproducible and comparable. An earlier unseeded comparison suggested a large gap between learning rates (73.9 vs 79.1%), but this turned out to be mostly random run-to-run variation rather than a real effect, after fixing the seed, the two learning rates perform alsomt identically:

Learning Rate           Best val Accuracy       Best Val loss       Stopped at epoch
1e-4                        78.1%                   0.65                7
5e-5                        77.7%                   0.63                9

Given the neglibile difference, lr = 1e-4 is used as the final ResNet18 configuration, since it reached itsbesy result in few epochs. The final model is saved as best_resnet18_Ir1e-4.pth.

Note: Even with a fixed seed, MPS training is not fully deterministic, so minor variations between runs may still occur.

### Project structure
```
project-root/
├── 01_preprocess.py                    # dataset loading, splitting, class weights
├── 02_dataset.py                       # Pytorch Dataset/Dataloader pipeline 
├── 03_train_resnet18.py                # ResNet18 training with checkpointing + early stopping
├── best_resnet18_Ir1e-4.pth            # Generated - best Renext 18 checkpint, not tracked in git. Also the one used.
├── best_resnet18_lr5e-5.pth
├── ham10000_processed_metadata.csv     # generated — not tracked in git
├── class_weights.csv                   # generated — not tracked in git
├── DataSet/                            # dataset — not tracked in git
├── .gitignore
└── README.md
```
### Next steps (not yet implented)
- [X] Build the pytorch dataset/dataloader pipeline
- [X] Implement the resnet18 baseline (transfer learning)
- [ ] Implement the vit-tiny model (transfer learning)
- [ ] Train and evaluate both modes
- [ ] Run experiments across 25 % 50% 75% 10% of training data
- [ ] Compare results using accuracy, precesion, recall, F1-score, and confusion matrices.
