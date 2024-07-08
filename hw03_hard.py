# -*- coding: utf-8 -*-
# %% Import necessary packages.
import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder

# This is for the progress bar.
from tqdm.auto import tqdm

#%%
# "cuda" only when GPUs are available.
device = "cuda:1" if torch.cuda.is_available() else "cpu"

# The number of training epochs.
n_epochs = 500

# Whether to do semi-supervised learning.
do_semi = True

# Batch size for training, validation, and testing.
# A greater batch size usually gives a more stable gradient.
# But the GPU memory is limited, so please adjust it carefully.
batch_size = 16

trainingEn = True
testingEn = True

#
loadModelPath = "models/hw03_hard/hw03_hard.ckpt_epoch_325.ckpt"
startEpoch = 325

# 
datasetPath = "./Datasets/food-11"
modelPath = "models/hw03_hard/"
modelName = "hw03_hard.ckpt"
lossesPath = "models/hw03_hard/loss&acc.csv"
predictionsPath = f"models/hw03_hard/{n_epochs}epochs_predict.csv"
#%%
"""## **Dataset, Data Loader, and Transforms**

Torchvision provides lots of useful utilities for image preprocessing, data wrapping as well as data augmentation.

Here, since our data are stored in folders by class labels, we can directly apply **torchvision.datasets.DatasetFolder** for wrapping data without much effort.

Please refer to [PyTorch official website](https://pytorch.org/vision/stable/transforms.html) for details about different transforms.
"""

# It is important to do data augmentation in training.
# However, not every augmentation is useful.
# Please think about what kind of augmentation is helpful for food recognition.
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((128, 128)),
    # You may add some transforms here.
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
])

# We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

augmentationTransforms = transforms.Compose([
    transforms.RandomAffine(degrees=(0, 360), translate=(0.125, 0.125), scale=(0.8, 1.2), shear=(16, 16)),
    transforms.ColorJitter()
])

#%%
# Construct datasets.
# The argument "loader" tells how torchvision reads the data.
train_set = DatasetFolder(os.path.join(datasetPath, "training/labeled"), loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
valid_set = DatasetFolder(os.path.join(datasetPath, "validation"), loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
unlabeled_set = DatasetFolder(os.path.join(datasetPath, "training/unlabeled"), loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
test_set = DatasetFolder(os.path.join(datasetPath, "testing"), loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)

# Construct data loaders.
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

#%% **Model**
# class Classifier(nn.Module):
#     def __init__(self):
#         super(Classifier, self).__init__()
#         # input image size: [3, 128, 128]
#         self.cnn_layer1 = nn.Sequential(
#             nn.Conv2d(3, 16, 3, 1, 1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),

#             nn.Conv2d(16, 32, 3, 1, 1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2, 0),
#         )
#         # image size: [32, 64, 64]
#         self.cnn_layer2 = nn.Sequential(
#             nn.Conv2d(32, 64, 3, 1, 1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),

#             nn.Conv2d(64, 64, 3, 1, 1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2, 0),
#         )
#         self.downsample2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0),
#             nn.BatchNorm2d(64),
#         )
#         # image size: [64, 32, 32]
#         self.cnn_layer3 = nn.Sequential(
#             nn.Conv2d(64, 128, 3, 1, 1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),

#             nn.Conv2d(128, 128, 3, 1, 1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2, 0),
#         )
#         self.downsample3 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0),
#             nn.BatchNorm2d(128),
#         )
#         # image size: [128, 16, 16]
#         self.cnn_layer4 = nn.Sequential(
#             nn.Conv2d(128, 256, 3, 1, 1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),

#             nn.Conv2d(256, 256, 3, 1, 1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2, 0),
#         )
#         self.downsample4 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0),
#             nn.BatchNorm2d(256),
#         )
#         # image size: [256, 8, 8]
#         self.cnn_layer5 = nn.Sequential(
#             nn.Conv2d(256, 256, 3, 1, 1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),

#             nn.Conv2d(256, 256, 3, 1, 1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2, 0),
#         )
#         self.downsample5 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=1, stride=2, padding=0),
#             nn.BatchNorm2d(256),
#         )
#         # image size: [256, 4, 4]

#         self.fc_layer1 = nn.Sequential(
#             nn.Linear(256 * 4 * 4, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Linear(64, 11)
#         )

#     def forward(self, x):
#         # input (x): [batch_size, 3, 128, 128]
#         # output: [batch_size, 11]

#         # Extract features by convolutional layers.
#         x = self.cnn_layer1(x)
#         x = self.cnn_layer2(x) + self.downsample2(x)
#         x = self.cnn_layer3(x) + self.downsample3(x)
#         x = self.cnn_layer4(x) + self.downsample4(x)
#         x = self.cnn_layer5(x) + self.downsample5(x)
#         # The extracted feature map must be flatten before going to fully-connected layers.
#         x = x.flatten(1)

#         # The features are transformed by fully-connected layers to obtain the final logits.
#         x = self.fc_layer1(x)
#         return x
    
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # The arguments for commonly used modules:
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        # input image size: [3, 128, 128]
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            # image size: [64, 64, 64]
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            # image size: [128, 32, 32]
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            # image size: [256, 16, 16]
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            # image size: [512, 8, 8]
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            # image size: [1024, 4, 4]
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            # image size: [1024, 2, 2]
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(1024*2*2, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(), 
            nn.Dropout(0.4),
            nn.Linear(256, 11)
        )

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]

        # Extract features by convolutional layers.
        x = self.cnn_layers(x)

        # The extracted feature map must be flatten before going to fully-connected layers.
        x = x.flatten(1)

        # The features are transformed by fully-connected layers to obtain the final logits.
        x = self.fc_layers(x)
        return x

#%% **Training**

def get_pseudo_labels(dataset, model, threshold=0.9):
    # This functions generates pseudo-labels of a dataset using given model.
    # It returns an instance of DatasetFolder containing images whose prediction confidences exceed a given threshold.
    # You are NOT allowed to use any models trained on external data for pseudo-labeling.

    # Construct a data loader.
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Make sure the model is in eval mode.
    model.eval()
    # Define softmax function.
    softmax = nn.Softmax(dim=-1)
    
    # Iterate over the dataset by batches.
    indices = []
    counter = 0
    newLabels = []
    for batch in tqdm(data_loader):
        img, _ = batch

        # Forward the data
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(img.to(device))

        # Obtain the probability distributions by applying softmax on logits.
        probs = softmax(logits)

        # ---------- TODO ----------
        # Filter the data and construct a new dataset.
        for prob in probs:
            newLabels.append(prob.argmax())
            if prob.max() > threshold and (prob>threshold*0.7).sum()<=1:
                indices.append(counter)
            counter = counter + 1
    for i, newLabel in enumerate(newLabels):
        dataset.samples[i] = (dataset.samples[i][0], newLabel.item())
    # # Turn off the eval mode.
    model.train()
    return Subset(dataset, indices)

# Initialize a model, and put it on the device specified.
model = Classifier().to(device)
model.device = device

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)
# %%
if trainingEn:
    # load model
    if loadModelPath is not None:
        print(f"loading model \"{loadModelPath}\"")
        checkpoint = torch.load( loadModelPath )
        model.load_state_dict(checkpoint["model_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_dict"])

    maxAcc = -1.0
    for epoch in range(startEpoch, n_epochs):
        # ---------- TODO ----------
        # In each epoch, relabel the unlabeled dataset for semi-supervised learning.
        # Then you can combine the labeled dataset and pseudo-labeled dataset for the training.
        if do_semi:
            # Obtain pseudo-labels for unlabeled data using trained model.
            pseudo_set = get_pseudo_labels(unlabeled_set, model)

            # Construct a new dataset and a data loader for training.
            # This is used in semi-supervised learning only.
            print("pseudo_set size : ", len(pseudo_set))
            concat_dataset = ConcatDataset([train_set, pseudo_set])
            train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        model.train()

        # These are used to record information in training.
        train_loss = []
        train_accs = []

        # Iterate the training set by batches.
        # for i, batch in enumerate(train_loader, 0):
        for batch in tqdm(train_loader):

            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            if len(labels) <= 1 :
                continue

            # Forward the data. (Make sure data and model are on the same device.)
            imgs = imgs.to(device)
            imgs = augmentationTransforms(imgs)
            logits = model(imgs)

            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss = criterion(logits, labels.to(device))

            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()

            # Compute the gradients for parameters.
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            # Update the parameters with computed gradients.
            optimizer.step()

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)

        # The average loss and accuracy of the training set is the average of the recorded values.
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()

        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []

        # Iterate the validation set by batches.
        for batch in tqdm(valid_loader):

            # A batch consists of image data and corresponding labels.
            imgs, labels = batch

            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs.to(device))

            # We can still compute the loss (but not the gradient).
            loss = criterion(logits, labels.to(device))

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        # Print the information.
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        # save model
        print("    saving models", end="")
        if maxAcc < valid_acc:
            maxAcc = valid_acc
            torch.save(model.state_dict(), os.path.join(modelPath, f"maxAcc.ckpt"))
        
        checkpointData ={   "model_dict" : model.state_dict(),
                            "optimizer_dict" : optimizer.state_dict(),
                        }
        checkpointName = os.path.join(modelPath, f"{modelName}_epoch_{epoch+1}.ckpt")
        torch.save(checkpointData, checkpointName)

        # save loss
        if (not os.path.exists(lossesPath)) or epoch==0:
            with open(lossesPath, mode="w", encoding="utf8") as csvFile:
                csvFile.write("epoch,train_loss,train_acc,valid_loss,valid_acc\n")
        with open(lossesPath, mode="a", encoding="utf8") as csvFile:
            csvFile.write(f"{epoch},{train_loss},{train_acc},{valid_loss},{valid_acc}\n")

        # delete part of model to save space
        print("    removing prev model", end="\n")
        checkpointName = os.path.join(modelPath, f"{modelName}_epoch_{epoch}.ckpt")
        if os.path.exists(checkpointName) and (epoch%10 != 0):
            os.remove(checkpointName)


#%% **Testing**

if testingEn:
    model.load_state_dict(torch.load(modelPath))
    # Make sure the model is in eval mode.
    # Some modules like Dropout or BatchNorm affect if the model is in training mode.
    model.eval()

    # Initialize a list to store the predictions.
    predictions = []

    # Iterate the testing set by batches.
    for batch in tqdm(test_loader):
        # A batch consists of image data and corresponding labels.
        # But here the variable "labels" is useless since we do not have the ground-truth.
        # If printing out the labels, you will find that it is always 0.
        # This is because the wrapper (DatasetFolder) returns images and labels for each batch,
        # so we have to create fake labels to make it work normally.
        imgs, labels = batch

        # We don't need gradient in testing, and we don't even have labels to compute loss.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # Take the class with greatest logit as prediction and record it.
        predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

    # Save predictions into the file.
    with open(predictionsPath, "w") as f:

        # The first row must be "Id, Category"
        f.write("Id,Category\n")

        # For the rest of the rows, each image id corresponds to a predicted class.
        for i, pred in  enumerate(predictions):
            f.write(f"{i},{pred}\n")