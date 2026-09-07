# Skin lesion Classification: CNN vs Vision Transformer

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
```
Learning Rate           Best val Accuracy       Best Val loss       Stopped at epoch
1e-4                        78.1%                   0.65                7
5e-5                        77.7%                   0.67                9
```
Given the neglibile difference, lr = 1e-4 is used as the final ResNet18 configuration, since it reached itsbesy result in few epochs. The final model is saved as best_resnet18_Ir1e-4.pth.

Note: Even with a fixed seed, MPS training is not fully deterministic, so minor variations between runs may still occur.

### Running the train ViT-Tiny

```bash
python 04_train_vit_tiny.py
```

**What it does**

Sample pipeline as 03_train_resnet18.py (class-weighted loss checkpointing, early stopping) but uses a pretrained ViT-Tiny model (timm's vit_tiny_patch16_224) instead of ResNet18, with a lower learning rate (3e-5) since transformers are more sensitive to aggressive updates during fine-tuning.

**note on hardware:** Vit_Tiny training triggered a RuntimeError on the MPS backend, caused by an internal reshape operation in the attention mechabism that isn't fully sypported by the current PyTorch/timm/MPS combination. Training runs correctly on CPU instead - this project use CPU for ViT-Tiny training as a result


**Results**
```
Model               Best Val Accuracy       Best val Loss           Stopped at epoch
RenNet18(lr = 1e4)  78.1%                   0.65                    7
ViT-Tiny(lr = 3e-5) 80.1%                   0.69                    19
```
ViT-Tiny outperformss ResNet19 by roughly 2 percentage points ob validation accuracy in this initial comparsion, though it requireds subsatantially more epochs and CPU time, to reach its best result. FInal model saved as best_vit_tiny_lr3e-4.pth

```
Model               Best Val Accuracy       Best val Loss           Stopped at epoch
ViT-Tiny(lr = 3e-5) 80.1%                   0.69                    19
ViT-Tiny(lr = 1e-5) 75.7%                   0.70                    9
```

Unlike the ResNet18 learning rate comparison (where the difference turned out to be mostly noise), the gap is meaningful: lr=1e-5 converged too slowly and tiggered early stopping before fully learning while lr=3e-5 had room to keep improving up to epoch 19. lr=3e-5 is used as the final ViT-Tiny configuration. Final model saved as best_vit_tiny_lr3e-4.pth.


### Project structure
```
project-root/
├── 01_preprocess.py                    # dataset loading, splitting, class weights
├── 02_dataset.py                       # Pytorch Dataset/Dataloader pipeline 
├── 03_train_resnet18.py                # ResNet18 training with checkpointing + early stopping
├── 04_train_vit_tiny.py                # Vit-Tiny trainin
├── best_resnet18_Ir1e-4.pth            # Generated - best Renext 18 checkpint, not tracked in git. Also the one used.
├── best_resnet18_lr5e-5.pth
├── best_vit_tiny_lr3e-4.pth            # generated - best Vit-tiny
├── best_vit_tiny_lr1e-5.pth            # generated - best Vit-tiny
├── ham10000_processed_metadata.csv     # generated — not tracked in git
├── class_weights.csv                   # generated — not tracked in git
├── DataSet/                            # dataset — not tracked in git
├── .gitignore
└── README.md
```
*Note all the different training dataset size paths are also saved jusy not tracked in git*


### Dataset size ablation
To investiate RQ2 (how the CNN and ViT performance gap changes with less training data), both models are retrained using 25%, 50%, 75% and 100% of the training set. Subest are sampled-per class (stratifled) with a fixed seed, so class propritons are presrved at every size.

**ResNet18 results**
```
Training data           Val accuracy
25%                     75.8%
50%                     74.0%
75%                     76.4%
100%                    78.1%
```
ResNet18 shos relatively small variation across dataste sizes (74-78&), with the performance at 25% already fairy close to the full-data results. This suggests the model's ImageNet pretraining is doing much of the heavly lifing, with fine-tuning data mainly providing incremental gains rather than being essential to reach reasonable performance. Note the 50% results is slightlt lower than 25%, with the dataset sizes this close together, some of this variation likely reflects with specific images were samples into each subest not just how many.

**ViT-Tiny results**
```
Training data           Val accuracy
25%                     67.6%
50%                     73.4%
75%                     79.1%
100%                    80.1%
```

Unlike ResNet18, ViT-Tiny shows a clear, steadily increasing trend as training data increases - substantially larger drop at 25% (67.6% vs ResNet18's 75.8%) that steadily closes as more data becomes available.

**Combined comparison**
```
Training data           ResNet18        ViT-Tiny        Gap(ViT- ResNet18)
25%                     75.8%           67.6%           -8.2
50%                     74.0%           73.4%           -0.6
75%                     76.4%           79.1%           +2.7
100%                    78.1%           80.1%           +2.0
```

This is the cnetral finding of the project, directly addressing both research question.
- RQ1 (Does ViT-Tiny outperform ResNet18?): yes, but only once enough training data is available - ViT only overtakes ResNet18 once training data reaches roughly 75% of the full set
- RQ2 (how does the performance gap change with training data size?): the gap reverses direction as data increases. WIth limited data (25%), ResNet18's ImageNet pretraining and built-in spatial inductive bias give it a clear advantage. As more data becomes available, ViT-Tiny's self-attention mechanism is able to learn effective representations and overtakes ResNet18, consistent with the data-efficiency concerns raised in the literature review

This supports the conclusion that Vision Transformer are a data-hungrier architecture: they need a larger training set to reach CNN-leve performance on this task, but are not necessarily the better choice when training data is scarce.

### Test Set evaluation
Both final models (trained on 100% of the training data) are evaluated once on the heldout test set (1481 images, never seen during training or hyperparameter tuning) via 05_evaluate.py.

**Overall test performance**
```
Model       Test Accuracy   Macro F1    Macro Prescision    Macro Recall
ResNet18    79.2%           0.644       0.603               0.721
ViT-Tiny    80.7%           0.689       0.667               0.721
```
ViT-tiny outperforms ResNet18 on both accuracy and Macro-averafed F1 on the test set. Notably, the macro F1 gap is proprtionally larger than the accuracy gap indicating ViT-Tiny handles the minority classes meaningfully better, not just the dominant nv class - this is exactly the kind of difference that raw accuracy alone would have hidden, justifying the macro-averaged metrics committed to in the Design section

**Per-Class F1-score**
```
Class       Support     ResNet18F1      ViT-Tiny f1
akiec       46          0.500           0.552
bcc         71          0.675           0.730
bk1         168         0.652           0.616
df          20          0.600           0.649
mel         165         0.555           0.579
nv          992         0.900           0.909
vasc        19          0.621           0.791
```
ViT-Tiny wins on 5 of 7 classes: ResNet18 is slightly better on bk1. The largest gap is on vasc, the rarest class (19 test images), where ViT-Tiny's F1 is notably higher (0.79 vs 0.62), driven mainly by better precision (few false positives).

**Confusion Matrices**

![Confusion Matrices for both Cnn And VIT](/CNN_vs_Transformer_for_Image_Classification/confusion_matrices.png)

The most clinically important error pattern in both models is confusion between mel(melanoma) and nv(benign nevi) - ResNet18 misclassifies 28 true melanoma cases as nv, and ViT-Tiny misclassifies 27. Given that melanoma is the most dangerous class in this dataset, this is a notable limitation for both models, and worth highligihting in the discussion: a real screening tool would need substantially better melanoma recall before being clinically usable, regardless of which architecture is used.


### Next steps (not yet implented)
- [X] Build the pytorch dataset/dataloader pipeline
- [X] Implement the resnet18 baseline (transfer learning)
- [X] Implement the vit-tiny model (transfer learning)
- [X] Run experiments across 25 % 50% 75% 10% of training data
- [X] Compare results using accuracy, precesion, recall, F1-score, and confusion matrices.


python 05_evaluate.py
